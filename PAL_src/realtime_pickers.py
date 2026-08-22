"""Picker adapters for the realtime AI-PAL workflow."""
import asyncio
import time

import torch
from obspy import UTCDateTime


class NativePickerAdapter(object):
    """Expose a native AI-PAL picker through the realtime interface."""

    def __init__(self, name, picker):
        self.name = name
        self.picker = picker
        self.device = picker.device

    def pick(self, stream, prepared=None):
        return self.picker.pick(
            stream, prepared=prepared, defer_waveform_qc=True
        )

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
        configured_npts = int(round(float(cfg.win_len) * float(cfg.samp_rate)))
        if configured_npts != 3000:
            raise ValueError(
                "PHN-SB requires win_len * samp_rate == 3000; got {} * {}"
                .format(cfg.win_len, cfg.samp_rate)
            )
        overlap_sec = float(cfg.overlap_sec)
        if overlap_sec < 0 or overlap_sec >= float(cfg.win_len):
            raise ValueError("PHN-SB overlap_sec must satisfy 0 <= overlap < win_len")
        overlap_samples = int(round(overlap_sec * float(cfg.samp_rate)))

        self.model = sbm.PhaseNet.from_pretrained(cfg.weights)
        model_rate = float(self.model.sampling_rate)
        if abs(model_rate - float(cfg.samp_rate)) > 1e-6:
            raise ValueError(
                "PHN-SB model sampling rate {} does not match configured {}"
                .format(model_rate, cfg.samp_rate)
            )
        self.model.to(self.device)
        self.model.eval()
        self.classify_args = {
            "P_threshold": cfg.trig_thres,
            "S_threshold": cfg.trig_thres,
            "batch_size": cfg.picker_batch_size,
            # SeisBench expects overlap in samples; case configs use seconds.
            "overlap": overlap_samples,
            "stacking": "avg",
            # ``pick`` already owns a private stream copy. Avoid another
            # hour-long copy inside SeisBench's annotation pipeline.
            "copy": False,
        }
        # SeisBench's synchronous classify() calls asyncio.run() each time.
        # In a station loop that repeatedly creates default-executor threads
        # and their native allocator arenas. Keep one loop/executor for the
        # lifetime of this adapter instead.
        self._event_loop = asyncio.new_event_loop()
        self.window_length_sec = float(cfg.win_len)
        self.window_stride_sec = self.window_length_sec - overlap_sec
        self.tp_dev = cfg.tp_dev
        self.ts_dev = cfg.ts_dev
        self.edge_exclusion_sec = float(cfg.taper_max_length_sec)
        self.pair_window_sec = self.window_length_sec

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
    def pick(self, stream, prepared=None):
        if len(stream) != 3:
            return []
        if prepared is not None:
            stream = prepared.stream.copy()
        else:
            stream = stream.copy()
            usable_start = stream[0].stats.starttime + self.edge_exclusion_sec
            usable_end = stream[0].stats.endtime - self.edge_exclusion_sec
            if usable_end <= usable_start:
                return []
            stream = stream.slice(usable_start, usable_end, nearest_sample=True)
        # The shared station preparation orders data E, N, Z. Give copied or
        # 1/2-component traces unique conventional IDs for SeisBench grouping.
        for trace, component in zip(stream, ("E", "N", "Z")):
            channel = trace.stats.channel or "HH?"
            trace.stats.channel = channel[:-1] + component

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self._event_loop.run_until_complete(
                self.model.classify_async(stream, **self.classify_args)
            )
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
            picks.append([net_sta, tp, ts, -1.0, p_prob, s_prob])
        print(
            "PHN-SB: {} P/S pairs from {} merged P + {} merged S "
            "({} raw phase picks) | {:.2f}s".format(
                len(picks), num_p, num_s, len(phase_picks),
                time.perf_counter() - t0
            )
        )
        return picks

    def close(self):
        """Close the persistent SeisBench async worker cleanly."""
        loop = getattr(self, "_event_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            shutdown_executor = getattr(loop, "shutdown_default_executor", None)
            if shutdown_executor is not None:
                loop.run_until_complete(shutdown_executor())
        finally:
            loop.close()
            self._event_loop = None
