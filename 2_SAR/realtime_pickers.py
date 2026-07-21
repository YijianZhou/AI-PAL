"""Picker adapters for the realtime AI-PAL workflow."""
import time

import numpy as np
import torch
from obspy import UTCDateTime


class SARPickerAdapter(object):
    """Expose the existing SAR picker through the common realtime interface."""

    name = "SAR"

    def __init__(self, sar_picker):
        self.picker = sar_picker

    def pick(self, stream):
        return self.picker.pick(stream)


class SeisBenchPhaseNetPicker(object):
    """Run SeisBench PhaseNet and convert independent phases to PAL P/S pairs."""

    name = "PHN-SB"

    def __init__(self, cfg, gpu_idx=0):
        try:
            import seisbench.models as sbm
        except ImportError as exc:
            raise ImportError(
                "PHN-SB is enabled, but seisbench is not importable"
            ) from exc

        self.device = torch.device(
            "cuda:{}".format(gpu_idx)
            if gpu_idx >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        self.model = sbm.PhaseNet.from_pretrained(cfg.phn_sb_weights)
        self.model.to(self.device)
        self.model.eval()
        self.classify_args = {
            "P_threshold": cfg.phn_sb_p_threshold,
            "S_threshold": cfg.phn_sb_s_threshold,
            "batch_size": cfg.picker_batch_size,
            "overlap": cfg.phn_sb_overlap,
            "stacking": "avg",
        }
        self.tp_dev = cfg.tp_dev
        self.ts_dev = cfg.ts_dev
        self.amp_win = cfg.amp_win
        self.pair_window_sec = (
            float(self.model.in_samples - 1) / float(self.model.sampling_rate)
        )
    @staticmethod
    def _phase_pick_values(pick):
        phase = str(getattr(pick, "phase", "")).upper()
        peak_time = getattr(pick, "peak_time", None)
        if peak_time is None:
            peak_time = getattr(pick, "start_time", None)
        probability = getattr(pick, "peak_value", -1.0)
        if peak_time is None or phase not in ("P", "S"):
            return None
        return phase, UTCDateTime(peak_time), float(probability)

    @staticmethod
    def _displacement_amplitude(stream, tp, ts, amp_win):
        window = stream.slice(tp - amp_win[0], ts + amp_win[1]).copy()
        if len(window) != 3:
            return -1.0
        npts = min(len(trace.data) for trace in window)
        if npts < 2:
            return -1.0
        displacement = []
        for trace in window:
            velocity = np.asarray(trace.data[:npts], dtype=np.float64)
            velocity = velocity - np.mean(velocity)
            displacement.append(np.cumsum(velocity) / trace.stats.sampling_rate)
        displacement = np.asarray(displacement)
        return float(np.sqrt(np.max(np.sum(displacement ** 2, axis=0))))

    @staticmethod
    def _merge_same_phase_picks(picks, tolerance_sec):
        """Keep the highest-probability pick within each time neighborhood."""
        merged = []
        for pick_time, probability in sorted(picks):
            match_idx = next(
                (
                    idx for idx, (kept_time, _) in enumerate(merged)
                    if abs(float(pick_time - kept_time)) < tolerance_sec
                ),
                None,
            )
            if match_idx is None:
                merged.append((pick_time, probability))
            elif probability > merged[match_idx][1]:
                merged[match_idx] = (pick_time, probability)
        return sorted(merged)

    def _pair_phases(self, phase_picks):
        p_picks = self._merge_same_phase_picks(
            [(time_i, prob) for phase, time_i, prob in phase_picks if phase == "P"],
            self.tp_dev,
        )
        s_picks = self._merge_same_phase_picks(
            [(time_i, prob) for phase, time_i, prob in phase_picks if phase == "S"],
            self.ts_dev,
        )
        pairs = [
            (tp, ts, p_prob, s_prob)
            for tp, p_prob in p_picks
            for ts, s_prob in s_picks
            if 0 < float(ts - tp) <= self.pair_window_sec
        ]
        return pairs, len(p_picks), len(s_picks)
    def pick(self, stream):
        if len(stream) != 3:
            return []
        stream = stream.copy()
        # The shared station preparation orders data E, N, Z. Give copied or
        # 1/2-component traces unique conventional IDs for SeisBench grouping.
        for trace, component in zip(stream, ("E", "N", "Z")):
            channel = trace.stats.channel or "HH?"
            trace.stats.channel = channel[:-1] + component

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.classify(stream, **self.classify_args)
        phase_picks = []
        for pick in getattr(output, "picks", []):
            values = self._phase_pick_values(pick)
            if values is not None:
                phase_picks.append(values)

        net_sta = "{}.{}".format(
            stream[0].stats.network, stream[0].stats.station
        )
        picks = []
        phase_pairs, num_p, num_s = self._pair_phases(phase_picks)
        for tp, ts, p_prob, s_prob in phase_pairs:
            s_amp = self._displacement_amplitude(
                stream, tp, ts, self.amp_win
            )
            picks.append([net_sta, tp, ts, s_amp, p_prob, s_prob])
        print(
            "PHN-SB: {} P/S pairs from {} merged P + {} merged S "
            "({} raw phase picks) | {:.2f}s".format(
                len(picks), num_p, num_s, len(phase_picks),
                time.perf_counter() - t0
            )
        )
        return picks
