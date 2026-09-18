"""Dual-group event repicking and PAL reassociation for AI-PAL."""

import copy
import importlib
import importlib.util
import csv
import ctypes
import gc
import hashlib
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
from obspy import Stream, UTCDateTime
from obspy.core.util.attribdict import AttribDict

import associator_pal
from phase_merge import (
    format_time, group_events, read_phase_file, resolve_pick_provenance,
)
from pick_ensemble import (
    format_picker_window_vote_ratios, repick_quality_code,
)
from waveform_qc import displacement_amplitude, is_glitch


POS_PICKER_REGISTRY = {
    "SAR": ("picker_SAR", "SARPositivePicker"),
    "FT": ("picker_FT", "FTPositivePicker"),
    "PHN": ("picker_PHN", "PHNPositivePicker"),
    "RUN": ("picker_RUN", "RUNPositivePicker"),
}

EVENT_REPICK_VERSION = "repick_quality_vote_ratio_v19"
EVENT_WAVEFORM_PRE_ORIGIN_SEC = 5.0
EVENT_WAVEFORM_POST_S_SEC = 10.0
EVENT_WAVEFORM_PLOT_DPI = 200
PAL_P_SNR_STA_SEC = 0.8
PAL_P_SNR_LTA_SEC = 6.0


class RetainedEventWaveform(object):
    """Small filtered window spilled until final publication."""

    def __init__(self, stream):
        self._stream = stream
        self._spill_path = None

    @property
    def stream(self):
        if self._stream is None and self._spill_path is not None:
            with open(self._spill_path, "rb") as fp:
                self._stream = pickle.load(fp)
        return self._stream

    def spill(self, path):
        path = os.path.abspath(path)
        partial = path + ".partial"
        with open(partial, "wb") as fp:
            pickle.dump(self._stream, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(partial, path)
        self._spill_path = path
        self._stream = None
        return self

    def release(self):
        self._stream = None
        if self._spill_path is not None:
            try:
                os.remove(self._spill_path)
            except FileNotFoundError:
                pass
            parent = os.path.dirname(self._spill_path)
            try:
                os.rmdir(parent)
            except OSError:
                pass
            self._spill_path = None


def _minimum_window_votes(cfg):
    """Scale the required repeated-window support with num_repeat."""
    return max(1, int(math.ceil(
        float(cfg.repick_num_repeat)
        * float(getattr(cfg, "repick_min_window_vote_ratio", 0.2))
    )))


def _trim_cpu_allocator():
    """Return released waveform arrays to Linux before repick batching."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _station_base(selector):
    return ".".join(str(selector).split(".")[:2])


def _distance_components_km(event, station):
    sta_lat, sta_lon, sta_ele = [float(value) for value in station[:3]]
    lat0 = 0.5 * (float(event["lat"]) + sta_lat)
    dx = 111.32 * (sta_lon - float(event["lon"])) * math.cos(
        math.radians(lat0)
    )
    dy = 111.32 * (sta_lat - float(event["lat"]))
    dz = float(event["depth"]) + sta_ele / 1000.0
    return dx, dy, dz


def _format_picker_uncertainties(values):
    fields = []
    for name in sorted(values):
        item = values[name]
        fields.append(
            "{}:tp={:.4f};ts={:.4f};pp={:.4f};sp={:.4f}".format(
                name,
                item["tp_std"],
                item["ts_std"],
                item["p_prob_std"],
                item["s_prob_std"],
            )
        )
    return "|".join(fields)


def _safe_event_filename(origin_time):
    origin_time = UTCDateTime(origin_time)
    return origin_time.strftime("%Y%m%dT%H%M%S.%fZ") + ".png"


def _safe_event_code(origin_time):
    return UTCDateTime(origin_time).strftime("%Y%m%dT%H%M%S.%fZ")


class PositivePickerAdapter(object):
    """Adapt one benchmark positive picker to a filtered ObsPy stream."""

    def __init__(self, name, module, picker, cfg):
        self.name = name
        self.module = module
        self.picker = picker
        self.cfg = cfg
        self.sampling_rate = float(module.samp_rate)
        self.num_channels = int(module.num_chn)
        self.window_seconds = float(module.win_len)

    def pick(
        self, stream, event_id, station, tp_pred, ts_pred, workflow_cfg,
    ):
        window_range = self._window_start_range(
            stream, tp_pred, ts_pred, workflow_cfg.repick_phase_buffer_sec
        )
        if window_range is None:
            return None
        start_low, start_high = window_range
        context_start = start_low
        context_end = start_high + self.window_seconds
        context = stream.slice(
            context_start, context_end, nearest_sample=True
        )
        if len(context) != self.num_channels:
            return None
        npts = min(len(trace.data) for trace in context)
        minimum_npts = int(round(self.window_seconds * self.sampling_rate))
        if npts < minimum_npts:
            return None
        data = np.asarray(
            [trace.data[:npts] for trace in context], dtype=np.float32
        )
        shard = np.zeros(
            (1, self.num_channels, npts + 2), dtype=np.float32
        )
        shard[0, :, 0] = float(tp_pred - context_start)
        shard[0, :, 1] = float(ts_pred - context_start)
        shard[0, :, 2:] = data

        low_rel = 0.0
        high_rel = max(0.0, float(start_high - context_start))
        self.module.configure_positive_picker(
            [low_rel, high_rel],
            workflow_cfg.repick_num_repeat,
            workflow_cfg.repick_batch_size,
            workflow_cfg.repick_random_seed,
            _minimum_window_votes(workflow_cfg),
        )
        meta = {
            "dataset": "AI-PAL-event-repick",
            "event_id": event_id,
            "trace_name": station,
            "station_key": station,
            "row_in_shard": 0,
            "p_rel_sec": float(tp_pred - context_start),
            "s_rel_sec": float(ts_pred - context_start),
        }
        rows = self.picker.pick_shard(shard, [meta])
        return self._select_pairs(
            rows,
            context_start,
            tp_pred,
            ts_pred,
            workflow_cfg.tp_dev,
            workflow_cfg.ts_dev,
        )

    @property
    def device(self):
        return self.picker.device

    def predict_preprocessed_batch(self, batch):
        """Run one already-normalized shared window batch."""
        with torch.inference_mode():
            if self.name == "SAR":
                logits = self.picker.model(
                    self.picker.window_batch_to_seq(batch)
                )
                return F.softmax(logits, dim=-1).detach().cpu().numpy()
            if self.name == "FT":
                amp_dtype = (
                    torch.bfloat16
                    if self.device.type == "cuda"
                    and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    logits = self.picker.model(batch)
                return F.softmax(logits.float(), dim=1).cpu().numpy()
            logits = self.picker.model(batch)
            return F.softmax(logits, dim=1).detach().cpu().numpy()

    def finalize_shared_votes(self, job, p_votes, s_votes, workflow_cfg):
        rows = self.picker.cluster_sample_votes(
            job["meta"], "P", p_votes, workflow_cfg.tp_dev
        )
        rows.extend(self.picker.cluster_sample_votes(
            job["meta"], "S", s_votes, workflow_cfg.ts_dev
        ))
        return self._select_pairs(
            rows,
            job["context_start"],
            job["tp_pred"],
            job["ts_pred"],
            workflow_cfg.tp_dev,
            workflow_cfg.ts_dev,
        )

    def _window_start_range(self, stream, tp_pred, ts_pred, buffer_seconds):
        phase_span = float(ts_pred - tp_pred)
        if phase_span <= 0 or phase_span >= self.window_seconds:
            return None
        if phase_span <= self.window_seconds - 2.0 * buffer_seconds:
            low = ts_pred + buffer_seconds - self.window_seconds
            high = tp_pred - buffer_seconds
        else:
            available_shift = self.window_seconds - phase_span
            low = tp_pred - available_shift
            high = tp_pred

        stream_start = max(trace.stats.starttime for trace in stream)
        stream_end = min(trace.stats.endtime for trace in stream)
        low = max(low, stream_start)
        high = min(high, stream_end - self.window_seconds)
        if high < low:
            return None
        return low, high

    def _select_pairs(
        self, rows, context_start, tp_pred, ts_pred, tp_dev, ts_dev,
    ):
        p_rows = [row for row in rows if row["phase"] == "P"]
        s_rows = [row for row in rows if row["phase"] == "S"]

        pairs = []
        for p_row in p_rows:
            tp = context_start + float(p_row["pick_time"])
            for s_row in s_rows:
                ts = context_start + float(s_row["pick_time"])
                if ts <= tp:
                    continue
                # The preliminary event contributes only its origin and
                # location. Keep every vote-supported pair and let PAL decide
                # which pairs form reassociated events.
                residual = abs(tp - tp_pred) / float(tp_dev)
                residual += abs(ts - ts_pred) / float(ts_dev)
                pairs.append((residual, tp, ts, p_row, s_row))
        if not pairs:
            return []
        selected = []
        for _, tp, ts, p_row, s_row in sorted(pairs, key=lambda item: item[0]):
            selected.append({
                "tp": tp,
                "ts": ts,
                "p_prob": float(p_row["pick_prob"]),
                "s_prob": float(s_row["pick_prob"]),
                "tp_std": float(p_row["pick_time_std"]),
                "ts_std": float(s_row["pick_time_std"]),
                "p_prob_std": float(p_row["pick_prob_std"]),
                "s_prob_std": float(s_row["pick_prob_std"]),
                "num_votes": min(
                    int(p_row["num_votes"]), int(s_row["num_votes"])
                ),
            })
        return selected

    # Kept for downstream developer scripts that exercised the old helper.
    def _select_pair(self, *args, **kwargs):
        pairs = self._select_pairs(*args, **kwargs)
        return pairs[0] if pairs else None


class EventRepicker(object):
    """Repick events with pos+neg and positive-only model ensembles."""

    def __init__(
        self, ai_pal_root, cfg, repicker_pos_neg_specs, station_file,
        station_dict=None, repicker_pos_specs=None,
        shared_continuous_pickers=None,
    ):
        self.ai_pal_root = Path(ai_pal_root)
        if str(self.ai_pal_root) not in sys.path:
            sys.path.insert(0, str(self.ai_pal_root))
        self.cfg = cfg
        self.repicker_specs = {
            "POS_NEG": dict(repicker_pos_neg_specs or {}),
            "POS": dict(repicker_pos_specs or {}),
        }
        shared_continuous_pickers = dict(shared_continuous_pickers or {})
        if any(
            name in shared_continuous_pickers for name in ("POS_NEG", "POS")
        ):
            self._shared_continuous_pickers = {
                group_name: dict(shared_continuous_pickers.get(group_name, {}))
                for group_name in ("POS_NEG", "POS")
            }
        else:
            self._shared_continuous_pickers = {
                "POS_NEG": shared_continuous_pickers,
                "POS": {},
            }
        self.station_file = Path(station_file)
        self.stations = (
            station_dict
            if station_dict is not None
            else cfg.get_sta_dict(str(self.station_file))
        )
        self.reassociation_params = self._resolve_reassociation_params()
        self.reassociator = associator_pal.PS_Pair_Assoc(
            self.stations, **self.reassociation_params
        )
        self.pickers = None
        repick_workers = max(1, int(getattr(
            self.cfg,
            "repick_num_workers",
            getattr(self.cfg, "num_workers", 1),
        )))
        self._window_executor = (
            ThreadPoolExecutor(
                max_workers=repick_workers,
                thread_name_prefix="repick-window",
            )
            if repick_workers > 1 else None
        )
        self._device_executor = None
        positive_values = {
            "repick_num_repeat": float(cfg.repick_num_repeat),
            "repick_batch_size": float(cfg.repick_batch_size),
            "repick_group_min_picker_support": float(
                cfg.repick_group_min_picker_support
            ),
            "tp_dev": float(cfg.tp_dev),
            "ts_dev": float(cfg.ts_dev),
            "vp": float(cfg.vp),
            "vs": float(cfg.vs),
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(
                "event repicker parameters must be positive: {}".format(
                    ", ".join(invalid)
                )
            )
        if float(cfg.repick_phase_buffer_sec) < 0:
            raise ValueError("repick_phase_buffer_sec must be nonnegative")
        vote_ratio = float(getattr(
            cfg, "repick_min_window_vote_ratio", 0.2
        ))
        if vote_ratio <= 0 or vote_ratio > 1:
            raise ValueError(
                "repick_min_window_vote_ratio must be in (0, 1]"
            )

    def set_station_geometry(self, stations):
        """Refresh epoch-aware station geometry before one interval."""
        if not stations:
            raise ValueError("event repicker station geometry is empty")
        self.stations = stations
        self.reassociator = associator_pal.PS_Pair_Assoc(
            self.stations, **self.reassociation_params
        )

    def _resolve_reassociation_params(self):
        """Resolve the full-network PAL parameters used after repicking."""
        configured = getattr(self.cfg, "subnet_assoc_params", {})
        params = dict(configured.get("default", {}))
        params.update(configured.get("full", {}))
        if "min_sta" not in params:
            raise KeyError("missing full-network min_sta association parameter")
        params["min_sta"] = int(params["min_sta"])
        if params["min_sta"] <= 0:
            raise ValueError("full-network min_sta must be positive")
        supported = {
            "xy_margin", "xy_grid", "z_grids", "vp", "ot_dev",
            "max_res", "max_drop", "min_sta", "lat_range", "lon_range",
        }
        params = {key: value for key, value in params.items() if key in supported}
        print(
            "post-repick full-network PAL reassociation: min_sta={}".format(
                params["min_sta"]
            ),
            flush=True,
        )
        return params

    def _load_pickers(self):
        if self.pickers is not None:
            return
        configured_names = {
            name
            for specs in self.repicker_specs.values()
            for name in specs
        }
        unknown = configured_names - set(POS_PICKER_REGISTRY)
        if unknown:
            raise KeyError("unsupported event repickers: {}".format(
                sorted(unknown)
            ))
        loaded = {}
        self.picker_groups = {"POS_NEG": {}, "POS": {}}
        for group_name, specs in self.repicker_specs.items():
            for name, spec in specs.items():
                key = "{}:{}".format(group_name, name)
                package, class_name = POS_PICKER_REGISTRY[name]
                config_path = Path(spec["config"])
                if not config_path.is_file():
                    raise FileNotFoundError(config_path)
                config_module_name = package + ".config"
                config_spec = importlib.util.spec_from_file_location(
                    config_module_name, config_path
                )
                config_module = importlib.util.module_from_spec(config_spec)
                sys.modules[config_module_name] = config_module
                try:
                    config_spec.loader.exec_module(config_module)
                except ImportError as exc:
                    legacy_message = (
                        "PAL_src/config_ai_pal.py was not found above"
                    )
                    if legacy_message not in str(exc):
                        raise
                    # Older case configs searched only their own ancestors for
                    # PAL_src. Case workdirs are intentionally independent of
                    # the installed AI-PAL tree, so retry with package context.
                    package_config = (
                        Path(__file__).resolve().parents[1]
                        / package / "config.py"
                    )
                    if not package_config.is_file():
                        raise
                    config_module = importlib.util.module_from_spec(config_spec)
                    config_module.__file__ = str(package_config)
                    sys.modules[config_module_name] = config_module
                    config_spec.loader.exec_module(config_module)
                    print(
                        "loaded legacy case config {} with AI-PAL package "
                        "context".format(config_path),
                        flush=True,
                    )
                package_module = importlib.import_module(package)
                setattr(package_module, "config", config_module)
                sys.modules.pop(package + ".picker_pos", None)
                importlib.invalidate_caches()
                module = importlib.import_module(package + ".picker_pos")
                picker_class = getattr(module, class_name)
                shared = self._shared_continuous_pickers.get(
                    group_name, {}
                ).get(name)
                if shared is not None:
                    shared = getattr(shared, "picker", shared)
                    picker = picker_class.__new__(picker_class)
                    picker.model = shared.model
                    picker.device = shared.device
                    load_message = "reused continuous model"
                else:
                    if "ckpt" not in spec:
                        raise ValueError(
                            "{} {} repicker requires an explicit ckpt file"
                            .format(group_name, name)
                        )
                    ckpt_path = Path(spec["ckpt"])
                    if not ckpt_path.is_file():
                        raise FileNotFoundError(ckpt_path)
                    picker = picker_class(
                        str(ckpt_path), -1, spec.get("gpu_idx", -1),
                    )
                    load_message = "loaded checkpoint"
                module.configure_positive_picker(
                    [0.0, 0.0], self.cfg.repick_num_repeat,
                    self.cfg.repick_batch_size, self.cfg.repick_random_seed,
                    _minimum_window_votes(self.cfg),
                )
                adapter = PositivePickerAdapter(
                    name, module, picker, module.cfg
                )
                loaded[key] = adapter
                self.picker_groups[group_name][name] = adapter
                print(
                    "{} {} repicker {} on {}".format(
                        load_message, group_name, name, picker.device
                    ),
                    flush=True,
                )
        window_lengths = {
            round(picker.window_seconds, 6) for picker in loaded.values()
        }
        sampling_rates = {
            round(picker.sampling_rate, 6) for picker in loaded.values()
        }
        if len(window_lengths) != 1 or len(sampling_rates) != 1:
            raise ValueError(
                "event repickers must share window length and sampling rate"
            )
        only_window = next(iter(window_lengths))
        only_rate = next(iter(sampling_rates))
        if only_window != round(float(self.cfg.win_len), 6):
            raise ValueError(
                "event-repicker win_len {} does not match AI-PAL {}".format(
                    only_window, self.cfg.win_len
                )
            )
        if only_rate != round(float(self.cfg.samp_rate), 6):
            raise ValueError(
                "event-repicker sampling rate {} does not match AI-PAL {}"
                .format(only_rate, self.cfg.samp_rate)
            )
        self.pickers = loaded
        print(
            "event repicker window support: {}/{} votes ({:.3f} ratio) | "
            "group support {}"
            .format(
                _minimum_window_votes(self.cfg),
                int(self.cfg.repick_num_repeat),
                float(getattr(
                    self.cfg, "repick_min_window_vote_ratio", 0.2
                )), int(self.cfg.repick_group_min_picker_support),
            ),
            flush=True,
        )

    def bind_continuous_pickers(self, pickers):
        """Reuse loaded pos+neg models selected for continuous inference."""
        if self.pickers is not None:
            raise RuntimeError("continuous models must be bound before loading")
        if any(name in pickers for name in ("POS_NEG", "POS")):
            for group_name in ("POS_NEG", "POS"):
                self._shared_continuous_pickers.setdefault(
                    group_name, {}
                ).update(pickers.get(group_name, {}))
        else:
            self._shared_continuous_pickers.setdefault(
                "POS_NEG", {}
            ).update(pickers)

    def load_pickers(self):
        """Load positive models before a large segment waveform is resident."""
        self._load_pickers()

    def close(self):
        """Close process-lifetime event-repicker executors."""
        if self._window_executor is not None:
            self._window_executor.shutdown(wait=True, cancel_futures=True)
            self._window_executor = None
        if self._device_executor is not None:
            self._device_executor.shutdown(wait=True, cancel_futures=True)
            self._device_executor = None
        _trim_cpu_allocator()

    def _prune_segment_waveforms(self, waveform_context, required_stations):
        """Drop segment-local stations that cannot participate in repicking."""
        required = set(required_stations)
        required_bases = {_station_base(station) for station in required}
        entries = self._waveform_entries(waveform_context)
        before = 0
        after = 0
        for entry in entries:
            waveforms = entry["waveforms"]
            before += len(waveforms)
            remove = [
                key for key in waveforms
                if key not in required and _station_base(key) not in required_bases
            ]
            for key in remove:
                holder = waveforms.pop(key, None)
                release = getattr(holder, "release", None)
                if release is not None:
                    release()
            after += len(waveforms)
        return before, after

    def process_hour(
        self, interval_start, interval_end, phase_path, catalog_path,
        waveform_context, association_summary,
    ):
        started = time.perf_counter()
        events = read_phase_file(phase_path)
        if not events:
            return {
                "enabled": True,
                "num_events": 0,
                "num_station_event_attempts": 0,
                "num_repicker_phase_pairs_generated": 0,
                "num_picks_reassociation_rejected": 0,
                "num_events_reassociated": 0,
                "num_events_reassociation_rejected": 0,
                "num_late_s_events_skipped": 0,
                "num_repick_windows": 0,
                "job_build_sec": 0.0,
                "window_prepare_sec": 0.0,
                "device_inference_wall_sec": 0.0,
                "device_transfer_sec": 0.0,
                "result_merge_sec": 0.0,
                "reassociation_sec": 0.0,
                "waveform_qc_measurement_sec": 0.0,
                "output_write_sec": 0.0,
                "plot_sec": 0.0,
                "num_event_plots": 0,
                "picker_seconds": {},
                "elapsed_sec": time.perf_counter() - started,
            }
        self._load_pickers()
        realtime_stream_end = association_summary.get("repick_stream_end")
        job_build_started = time.perf_counter()
        event_station_jobs, num_late_s_events = self._build_jobs(
            events, realtime_stream_end=realtime_stream_end
        )
        waveform_count_before = None
        waveform_count_after = None
        if association_summary.get("segment_local_waveforms", False):
            waveform_count_before, waveform_count_after = (
                self._prune_segment_waveforms(
                    waveform_context, event_station_jobs
                )
            )
            _trim_cpu_allocator()
            print(
                "realtime repick waveform pruning: {} -> {} stations"
                .format(waveform_count_before, waveform_count_after),
                flush=True,
            )
        candidates = self._build_candidate_jobs(
            events, event_station_jobs, waveform_context
        )
        amplitude_streams = {}
        for candidate in candidates:
            station = candidate["station"]
            amplitude_streams[station] = candidate["stream"]
            geometry_station = self._station_geometry_key(station)
            if geometry_station is not None:
                amplitude_streams.setdefault(
                    geometry_station, candidate["stream"]
                )
        # Initial phase rows define only the adaptive station-distance extent.
        # Every reassociation input below must come from the eight repickers.
        for event in events:
            event["picks"] = []
        job_build_sec = time.perf_counter() - job_build_started
        num_attempts = len(candidates)
        num_generated = 0
        num_glitch_rejected = 0
        num_picks_reassociation_rejected = 0
        num_events_reassociated = 0
        num_events_reassociation_rejected = 0
        picker_seconds = {name: 0.0 for name in self.pickers}
        window_prepare_sec = 0.0
        device_inference_wall_sec = 0.0
        device_transfer_sec = 0.0
        result_merge_sec = 0.0
        num_windows = 0
        chunk_size = max(
            1,
            int(self.cfg.repick_batch_size) * 4
            // int(self.cfg.repick_num_repeat),
        )
        for chunk_start in range(0, len(candidates), chunk_size):
            candidate_chunk = candidates[chunk_start:chunk_start + chunk_size]
            prepare_started = time.perf_counter()
            prepared_jobs = self._prepare_job_chunk(candidate_chunk)
            window_prepare_sec += time.perf_counter() - prepare_started
            if not prepared_jobs:
                continue
            windows, owners, window_starts = self._flatten_prepared_jobs(
                prepared_jobs
            )
            num_windows += len(windows)
            inference_started = time.perf_counter()
            per_picker, chunk_picker_seconds, transfer_seconds = (
                self._run_picker_device_groups(
                    windows, owners, window_starts, prepared_jobs
                )
            )
            device_inference_wall_sec += time.perf_counter() - inference_started
            device_transfer_sec += transfer_seconds
            for name, seconds in chunk_picker_seconds.items():
                picker_seconds[name] += seconds

            merge_started = time.perf_counter()
            for job_index, job in enumerate(prepared_jobs):
                picker_results = {
                    name: results[job_index]
                    for name, results in per_picker.items()
                    if results[job_index] is not None
                }
                group_results = self._cluster_repicker_groups(
                    picker_results, job["tp_pred"], job["ts_pred"]
                )
                results = self._combine_repicker_groups(job, group_results)
                if not results:
                    continue
                if bool(getattr(self.cfg, "rm_glitch", True)):
                    accepted_results = []
                    for result in results:
                        if is_glitch(
                            job["stream"],
                            UTCDateTime(result["p"]),
                            UTCDateTime(result["s"]),
                            self.cfg,
                        ):
                            num_glitch_rejected += 1
                        else:
                            accepted_results.append(result)
                    results = accepted_results
                if not results:
                    continue
                event = events[job["event_index"]]
                num_generated += len(results)
                event["picks"].extend(results)
            result_merge_sec += time.perf_counter() - merge_started

        reassociation_started = time.perf_counter()
        waveform_qc_measurement_sec = 0.0
        reassociated_events = []
        for event in events:
            reassociated = self._reassociate_event(event)
            if not reassociated:
                num_events_reassociation_rejected += 1
                num_picks_reassociation_rejected += len(event["picks"])
                continue
            selected_pick_ids = {
                id(pick)
                for candidate in reassociated
                for pick in candidate["picks"]
            }
            num_picks_reassociation_rejected += sum(
                id(pick) not in selected_pick_ids for pick in event["picks"]
            )
            waveform_qc_started = time.perf_counter()
            for candidate in reassociated:
                self._measure_final_event_waveform_qc(
                    candidate, amplitude_streams
                )
            waveform_qc_measurement_sec += (
                time.perf_counter() - waveform_qc_started
            )
            reassociated_events.extend(reassociated)
            num_events_reassociated += len(reassociated)

        events = reassociated_events
        reassociation_sec = max(
            0.0,
            time.perf_counter() - reassociation_started
            - waveform_qc_measurement_sec,
        )
        provenance_counts = {
            "both_groups": 0,
            "pos_neg_only": 0,
            "pos_only": 0,
        }
        for event in events:
            for pick in event["picks"]:
                provenance = resolve_pick_provenance([
                    pick.get("pick_provenance", "initial")
                ])
                if provenance not in provenance_counts:
                    raise RuntimeError(
                        "postprocessed phase is not repicker-derived: {}"
                        .format(provenance)
                    )
                provenance_counts[provenance] += 1

        output_started = time.perf_counter()
        self._write_outputs(events, phase_path, catalog_path)
        output_write_sec = time.perf_counter() - output_started
        snapshot_started = time.perf_counter()
        event_waveform_snapshots = []
        if bool(association_summary.get("defer_event_waveform_plot", False)):
            # Keep only short, final-event waveform windows in memory. The
            # coordinator plots them after corrected segment finalization.
            final_output_events = read_phase_file(phase_path)
            event_waveform_snapshots = self._capture_event_waveforms(
                final_output_events, waveform_context
            )
        waveform_snapshot_sec = time.perf_counter() - snapshot_started
        elapsed = time.perf_counter() - started
        summary = {
            "enabled": True,
            "num_events": len(events),
            "num_station_event_attempts": num_attempts,
            "num_repicker_phase_pairs_generated": num_generated,
            "num_repicker_phase_pairs_glitch_rejected": num_glitch_rejected,
            "num_picks_reassociation_rejected": (
                num_picks_reassociation_rejected
            ),
            "num_events_reassociated": num_events_reassociated,
            "num_events_reassociation_rejected": (
                num_events_reassociation_rejected
            ),
            "num_phase_pairs_both_groups": provenance_counts["both_groups"],
            "num_phase_pairs_pos_neg_only": provenance_counts["pos_neg_only"],
            "num_phase_pairs_pos_only": provenance_counts["pos_only"],
            "num_late_s_events_skipped": num_late_s_events,
            "num_repick_windows": num_windows,
            "job_build_sec": job_build_sec,
            "window_prepare_sec": window_prepare_sec,
            "device_inference_wall_sec": device_inference_wall_sec,
            "device_transfer_sec": device_transfer_sec,
            "result_merge_sec": result_merge_sec,
            "reassociation_sec": reassociation_sec,
            "waveform_qc_measurement_sec": waveform_qc_measurement_sec,
            "output_write_sec": output_write_sec,
            "plot_sec": 0.0,
            "waveform_snapshot_sec": waveform_snapshot_sec,
            "num_event_plots": 0,
            "num_waveform_stations_before_prune": waveform_count_before,
            "num_waveform_stations_after_prune": waveform_count_after,
            "picker_seconds": picker_seconds,
            "_event_waveform_snapshots": event_waveform_snapshots,
            "elapsed_sec": elapsed,
        }
        interval_label = (
            str(interval_start)
            if interval_start == interval_end
            else "{} -- {}".format(interval_start, interval_end)
        )
        print(
            "event repicking complete: {} | {} events | "
            "{} station-event jobs | {} shared windows | {} repicker pairs | "
            "{} glitch pairs rejected | "
            "{} picks rejected by reassociation | "
            "{} events reassociated | {} rejected by reassociation | "
            "{} late-S events skipped | device inference {:.2f}s | "
            "total {:.2f}s".format(
                interval_label,
                len(events),
                num_attempts,
                num_windows,
                num_generated,
                num_glitch_rejected,
                num_picks_reassociation_rejected,
                num_events_reassociated,
                num_events_reassociation_rejected,
                num_late_s_events,
                device_inference_wall_sec,
                elapsed,
            ),
            flush=True,
        )
        return summary

    def process_event(
        self, event_time, phase_path, catalog_path, waveform_context,
        association_summary=None,
    ):
        """Repick and reassociate one initial event."""
        return self.process_hour(
            event_time,
            event_time,
            phase_path,
            catalog_path,
            waveform_context,
            association_summary or {},
        )

    def waveform_requests(self, events):
        """Return exact station spans needed by event repicking and plotting."""
        if not events:
            return {}
        jobs, _ = self._build_jobs(copy.deepcopy(events))
        window_seconds = (
            next(iter(self.pickers.values())).window_seconds
            if self.pickers else float(self.cfg.win_len)
        )
        requests = {}
        for station, event_indices in jobs.items():
            bounds = []
            for event_index in event_indices:
                event = events[event_index]
                tp_pred, ts_pred = self._predicted_phases(event, station)
                bounds.append((
                    tp_pred - window_seconds,
                    ts_pred + window_seconds,
                ))
            if bounds:
                requests[station] = (
                    min(value[0] for value in bounds),
                    max(value[1] for value in bounds),
                )
        return requests

    def write_events(self, events, phase_path, catalog_path=None):
        """Write phase/catalog rows using the canonical extended schema."""
        self._write_outputs(events, phase_path, catalog_path)

    def qc_initial_events(
        self, phase_path, waveform_context, catalog_path=None,
    ):
        """Measure and glitch-check only picks used by initial detections."""
        events = read_phase_file(phase_path)
        rejected_picks = 0
        params = dict(getattr(
            self.cfg, "subnet_assoc_params", {}
        ).get("default", {}))
        params.update(getattr(
            self.cfg, "subnet_assoc_params", {}
        ).get("full", {}))
        min_sta = int(params.get("min_sta", 4))
        accepted = []
        for event in events:
            picks = []
            for pick in event["picks"]:
                station = self._station_geometry_key(pick["sta"])
                if station is None:
                    continue
                stream = self._merged_station_stream(
                    waveform_context,
                    station,
                    start_time=UTCDateTime(pick["p"]) - float(self.cfg.amp_win[0]),
                    end_time=UTCDateTime(pick["s"]) + float(self.cfg.amp_win[1]),
                )
                if stream is not None and bool(getattr(
                    self.cfg, "rm_glitch", True
                )) and is_glitch(
                    stream, UTCDateTime(pick["p"]),
                    UTCDateTime(pick["s"]), self.cfg,
                ):
                    rejected_picks += 1
                    continue
                pick["score"] = (
                    self._displacement_amplitude(
                        stream, UTCDateTime(pick["p"]), UTCDateTime(pick["s"])
                    ) if stream is not None else -1.0
                )
                picks.append(pick)
            event["picks"] = picks
            if len({_station_base(pick["sta"]) for pick in picks}) < min_sta:
                continue
            amplitude_rows = np.asarray([
                (self._station_geometry_key(pick["sta"]), pick["score"])
                for pick in picks
                if self._station_geometry_key(pick["sta"]) is not None
            ], dtype=np.dtype([
                ("net_sta", object), ("s_amp", np.float64),
            ]))
            event_location = self.reassociator.calc_mag(amplitude_rows, {
                "evt_ot": UTCDateTime(event["time"]),
                "evt_lat": float(event["lat"]),
                "evt_lon": float(event["lon"]),
                "evt_dep": float(event["depth"]),
                "mag": -1.0,
            })
            event["mag"] = float(event_location.get("mag", -1.0))
            accepted.append(event)
        self._write_outputs(accepted, phase_path, catalog_path)
        return {
            "num_input_events": len(events),
            "num_output_events": len(accepted),
            "num_rejected_events": len(events) - len(accepted),
            "num_rejected_picks": rejected_picks,
        }

    def plot_events(self, events, waveform_context, output_dir):
        """Publish plots for already reassociated and merged events."""
        return self._plot_events(events, waveform_context, Path(output_dir))

    def save_filtered_event_waveforms(
        self, events, waveform_context, output_dir,
    ):
        """Write final filtered event snippets as relocation-ready SAC files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        num_events = 0
        num_station_streams = 0
        for event in events:
            origin = UTCDateTime(event["time"])
            event_start = origin - EVENT_WAVEFORM_PRE_ORIGIN_SEC
            event_end = max(
                [UTCDateTime(pick["s"]) for pick in event["picks"]],
                default=origin + 50.0,
            ) + EVENT_WAVEFORM_POST_S_SEC
            event_dir = output_dir / _safe_event_code(origin)
            event_dir.mkdir(parents=True, exist_ok=True)
            wrote_event = False
            for pick in event["picks"]:
                geometry_key = self._station_geometry_key(pick["sta"])
                if geometry_key is None:
                    continue
                stream = self._merged_station_stream(
                    waveform_context,
                    geometry_key,
                    start_time=event_start,
                    end_time=event_end,
                )
                if stream is None or len(stream) != int(self.cfg.num_chn):
                    continue
                clipped = stream.slice(
                    event_start, event_end, nearest_sample=True
                ).copy()
                if len(clipped) != int(self.cfg.num_chn):
                    continue
                station_code = re.sub(
                    r"[^A-Za-z0-9_.-]+", "_", pick["sta"]
                )
                for channel_index, trace in enumerate(clipped):
                    trace.data = np.asarray(trace.data, dtype=np.float32)
                    # Reset inherited daily-file SAC timing headers so the
                    # snippet reference is its own start time (b=0).
                    trace.stats.sac = AttribDict()
                    trace.stats.sac.o = float(origin - trace.stats.starttime)
                    trace.stats.sac.t0 = float(
                        UTCDateTime(pick["p"]) - trace.stats.starttime
                    )
                    trace.stats.sac.t1 = float(
                        UTCDateTime(pick["s"]) - trace.stats.starttime
                    )
                    output_path = event_dir / "{}.{}.sac".format(
                        station_code, channel_index
                    )
                    partial_path = output_path.with_suffix(".partial")
                    trace.write(str(partial_path), format="SAC")
                    partial_path.replace(output_path)
                num_station_streams += 1
                wrote_event = True
            num_events += int(wrote_event)
            callback = getattr(
                self.cfg, "event_waveform_complete_callback", None
            )
            if wrote_event and callback is not None:
                callback(origin, event_dir)
        print(
            "filtered event waveforms: {} events | {} station streams | {}"
            .format(num_events, num_station_streams, output_dir),
            flush=True,
        )
        return {
            "num_waveform_events": num_events,
            "num_waveform_station_streams": num_station_streams,
        }

    def capture_event_waveforms(self, events, waveform_context):
        """Retain compact filtered windows for deferred final products."""
        return self._capture_event_waveforms(events, waveform_context)

    def _station_geometry_key(self, station):
        if station in self.stations:
            return station
        base = _station_base(station)
        candidates = sorted(
            key for key in self.stations if _station_base(key) == base
        )
        return candidates[0] if candidates else None

    def _plot_events(self, events, waveform_context, output_dir):
        """Plot one final event per PNG from retained filtered waveforms."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        output_dir.mkdir(parents=True, exist_ok=True)
        provenance_colors = {
            "both_groups": "#2878B5",
            "pos_neg_only": "#6F4E9C",
            "pos_only": "#3A923A",
            "initial": "#CC741D",
        }
        plotted = 0
        for event in events:
            origin = UTCDateTime(event["time"])
            event_end = max(
                [UTCDateTime(pick["s"]) for pick in event["picks"]],
                default=origin + 50.0,
            ) + EVENT_WAVEFORM_POST_S_SEC
            event_start = origin - EVENT_WAVEFORM_PRE_ORIGIN_SEC
            picks_by_station = {}
            for pick in event["picks"]:
                geometry_key = self._station_geometry_key(pick["sta"])
                if geometry_key is None:
                    continue
                picks_by_station.setdefault(geometry_key, []).append(pick)
            station_rows = []
            for geometry_key, station_picks in picks_by_station.items():
                dx, dy, _ = _distance_components_km(
                    event, self.stations[geometry_key]
                )
                stream = self._merged_station_stream(
                    waveform_context,
                    geometry_key,
                    start_time=event_start,
                    end_time=event_end,
                )
                if stream is None or len(stream) == 0:
                    continue
                station_rows.append((
                    math.hypot(dx, dy), geometry_key, station_picks, stream,
                ))
            station_rows.sort(key=lambda row: (row[0], row[1]))
            if not station_rows:
                continue

            # Keep a fixed physical height per station so a component has the
            # same visual scale in sparse and dense event plots.  Each station
            # owns three one-unit component lanes.
            figure_height = max(5.0, 2.2 + 0.72 * len(station_rows))
            fig, ax = plt.subplots(figsize=(14, figure_height))
            y_ticks = []
            y_labels = []
            num_waveform_lines = 0
            for station_index, (
                distance_km, station, station_picks, stream,
            ) in enumerate(
                station_rows
            ):
                station_base = 3.0 * station_index
                y_ticks.append(station_base + 1.0)
                y_labels.append("{}  {:.1f} km".format(
                    station, distance_km
                ))
                channel_shades = ("#555555", "#888888", "#B0B0B0")
                for channel_index, trace in enumerate(stream[:3]):
                    sampling_rate = float(trace.stats.sampling_rate)
                    # Convert absolute UTC bounds to sample indices explicitly.
                    # This avoids relying on plotting libraries to interpret
                    # sample number as seconds and preserves non-100-Hz inputs.
                    index_start = max(
                        0,
                        int(math.ceil(
                            float(event_start - trace.stats.starttime)
                            * sampling_rate
                        )),
                    )
                    index_end = min(
                        len(trace.data),
                        int(math.floor(
                            float(event_end - trace.stats.starttime)
                            * sampling_rate
                        )) + 1,
                    )
                    if index_end <= index_start:
                        continue
                    data = np.ma.filled(
                        trace.data[index_start:index_end], np.nan
                    ).astype(np.float32, copy=False)
                    if len(data) == 0:
                        continue
                    finite = np.isfinite(data)
                    if not np.any(finite):
                        continue
                    center = float(np.median(data[finite]))
                    centered = data - center
                    amplitude = float(np.percentile(
                        np.abs(centered[finite]), 99.5
                    ))
                    if not np.isfinite(amplitude) or amplitude <= 0:
                        amplitude = float(np.max(np.abs(centered[finite])))
                    if not np.isfinite(amplitude) or amplitude <= 0:
                        continue
                    normalized = np.full_like(data, np.nan)
                    normalized[finite] = np.clip(
                        centered[finite] / amplitude, -1.0, 1.0
                    )
                    first_time = (
                        trace.stats.starttime
                        + index_start / sampling_rate
                    )
                    sample_times = (
                        float(first_time - origin)
                        + np.arange(len(data), dtype=np.float32)
                        / sampling_rate
                    )
                    component_baseline = station_base + channel_index
                    ax.plot(
                        sample_times,
                        component_baseline + 0.42 * normalized,
                        color=channel_shades[channel_index],
                        linewidth=0.55,
                        alpha=0.95,
                        zorder=2,
                    )
                    num_waveform_lines += 1

                for pick in station_picks:
                    provenance = resolve_pick_provenance([
                        pick.get("pick_provenance", "initial")
                    ])
                    color = provenance_colors.get(
                        provenance, "#222222"
                    )
                    p_relative = float(UTCDateTime(pick["p"]) - origin)
                    s_relative = float(UTCDateTime(pick["s"]) - origin)
                    ax.vlines(
                        p_relative, station_base - 0.48, station_base + 2.48,
                        color=color, linewidth=1.4, linestyle="-", zorder=4,
                    )
                    ax.vlines(
                        s_relative, station_base - 0.48, station_base + 2.48,
                        color=color, linewidth=1.4, linestyle="--", zorder=4,
                    )

            ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_xlim(float(event_start - origin), float(event_end - origin))
            ax.set_ylim(-0.7, 3.0 * len(station_rows) - 0.3)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_xlabel("Time relative to origin (s)")
            ax.set_ylabel("Stations ordered by epicentral distance")
            ax.set_title(
                "{} | M {:.2f} | {:.5f}, {:.5f} | depth {:.1f} km".format(
                    str(origin), event["mag"], event["lat"], event["lon"],
                    event["depth"],
                )
            )
            ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
            legend = [
                Line2D([0], [0], color=color, lw=2, label=label)
                for label, color in (
                    ("POS_NEG + POS", provenance_colors["both_groups"]),
                    ("POS_NEG only", provenance_colors["pos_neg_only"]),
                    ("POS only", provenance_colors["pos_only"]),
                    ("Initial/reference", provenance_colors["initial"]),
                )
            ]
            legend.extend([
                Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="P"),
                Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="S"),
            ])
            ax.legend(handles=legend, loc="upper right", frameon=False, ncol=6)
            fig.tight_layout()
            output_path = output_dir / _safe_event_filename(origin)
            partial_path = output_path.with_name(
                output_path.stem + ".partial.png"
            )
            fig.savefig(
                partial_path,
                dpi=EVENT_WAVEFORM_PLOT_DPI,
                bbox_inches="tight",
            )
            plt.close(fig)
            partial_path.replace(output_path)
            plotted += 1
            if num_waveform_lines == 0:
                print(
                    "warning: event plot {} contains pick markers but no "
                    "finite nonzero waveform samples".format(output_path),
                    flush=True,
                )
        print(
            "event waveform plots: {} PNG files in {}".format(
                plotted, output_dir
            ),
            flush=True,
        )
        return plotted

    def _capture_event_waveforms(self, events, waveform_context):
        """Spill compact filtered windows for deferred final-event plotting."""
        snapshots = []
        cache_root = os.path.join(
            self.cfg.out_root, "_internal", "event_waveform_cache"
        )
        os.makedirs(cache_root, exist_ok=True)
        for event in events:
            origin = UTCDateTime(event["time"])
            event_end = max(
                [UTCDateTime(pick["s"]) for pick in event["picks"]],
                default=origin + 50.0,
            ) + EVENT_WAVEFORM_POST_S_SEC
            event_start = origin - EVENT_WAVEFORM_PRE_ORIGIN_SEC
            waveforms = {}
            event_cache = tempfile.mkdtemp(prefix="event_", dir=cache_root)
            for pick in event["picks"]:
                geometry_key = self._station_geometry_key(pick["sta"])
                if geometry_key is None or geometry_key in waveforms:
                    continue
                stream = self._merged_station_stream(
                    waveform_context,
                    geometry_key,
                    start_time=event_start,
                    end_time=event_end,
                )
                if stream is None or len(stream) == 0:
                    continue
                clipped = stream.slice(
                    event_start, event_end, nearest_sample=True
                ).copy()
                if clipped:
                    filename = re.sub(
                        r"[^A-Za-z0-9_.-]+", "_", geometry_key
                    ) + ".pkl"
                    waveforms[geometry_key] = RetainedEventWaveform(
                        clipped
                    ).spill(os.path.join(event_cache, filename))
            if not waveforms:
                try:
                    os.rmdir(event_cache)
                except OSError:
                    pass
            snapshots.append({
                "event": copy.deepcopy(event),
                "start": event_start,
                "end": event_end,
                "waveforms": waveforms,
            })
        return snapshots

    def _build_candidate_jobs(
        self, events, event_station_jobs, waveform_context,
    ):
        candidates = []
        for station, event_indices in sorted(event_station_jobs.items()):
            station_jobs = []
            for event_index in event_indices:
                event = events[event_index]
                epicentral_distance_km = self._epicentral_distance_km(
                    event, station
                )
                tp_pred, ts_pred = self._predicted_phases(event, station)
                station_jobs.append((
                    event_index, event, tp_pred, ts_pred,
                    epicentral_distance_km,
                ))
            if not station_jobs:
                continue
            window_seconds = next(iter(self.pickers.values())).window_seconds
            needed_start = min(
                job[2] for job in station_jobs
            ) - window_seconds
            needed_end = max(
                job[3] for job in station_jobs
            ) + window_seconds
            stream = self._merged_station_stream(
                waveform_context,
                station,
                start_time=needed_start,
                end_time=needed_end,
            )
            if stream is None:
                continue
            for (
                event_index, event, tp_pred, ts_pred,
                epicentral_distance_km,
            ) in station_jobs:
                candidates.append({
                    "stream": stream,
                    "event": event,
                    "event_index": event_index,
                    "event_id": "{}|{}".format(
                        event["time"].isoformat(), event_index
                    ),
                    "station": station,
                    "tp_pred": tp_pred,
                    "ts_pred": ts_pred,
                    "epicentral_distance_km": epicentral_distance_km,
                })
        return candidates

    def _prepare_job_chunk(self, candidates):
        if self._window_executor is None or len(candidates) == 1:
            prepared = [self._prepare_shared_job(job) for job in candidates]
        else:
            prepared = list(self._window_executor.map(
                self._prepare_shared_job, candidates
            ))
        return [job for job in prepared if job is not None]

    def _prepare_shared_job(self, candidate):
        template = next(iter(self.pickers.values()))
        window_range = template._window_start_range(
            candidate["stream"],
            candidate["tp_pred"],
            candidate["ts_pred"],
            self.cfg.repick_phase_buffer_sec,
        )
        if window_range is None:
            return None
        start_low, start_high = window_range
        context_start = start_low
        context_end = start_high + template.window_seconds
        context = candidate["stream"].slice(
            context_start, context_end, nearest_sample=True
        )
        if len(context) != template.num_channels:
            return None
        npts = min(len(trace.data) for trace in context)
        window_npts = int(round(
            template.window_seconds * template.sampling_rate
        ))
        if npts < window_npts:
            return None
        data = np.asarray(
            [trace.data[:npts] for trace in context], dtype=np.float32
        )
        high_rel = max(0.0, float(start_high - context_start))
        starts = self._shared_random_starts(candidate, high_rel)
        windows = []
        actual_starts = []
        for start_sec in starts:
            start_index = int(round(start_sec * template.sampling_rate))
            end_index = start_index + window_npts
            if start_index < 0 or end_index > npts:
                continue
            windows.append(data[:, start_index:end_index])
            actual_starts.append(start_index / template.sampling_rate)
        if not windows:
            return None
        windows = np.stack(windows).astype(np.float32, copy=False)
        windows -= np.mean(windows, axis=2, keepdims=True)
        if bool(self.cfg.global_max_norm):
            scale = np.max(np.abs(windows), axis=(1, 2), keepdims=True)
        else:
            scale = np.max(np.abs(windows), axis=2, keepdims=True)
        windows /= np.maximum(scale, np.float32(1e-12))
        prepared = dict(candidate)
        prepared.update({
            "context_start": context_start,
            "windows": np.ascontiguousarray(windows),
            "window_starts": np.asarray(actual_starts, dtype=np.float32),
            "meta": {
                "dataset": "AI-PAL-event-repick",
                "event_id": candidate["event_id"],
                "trace_name": candidate["station"],
                "station_key": candidate["station"],
                "row_in_shard": 0,
                "p_rel_sec": float(candidate["tp_pred"] - context_start),
                "s_rel_sec": float(candidate["ts_pred"] - context_start),
            },
        })
        return prepared

    def _shared_random_starts(self, candidate, high_rel):
        repeats = int(self.cfg.repick_num_repeat)
        if repeats <= 1:
            return np.asarray([high_rel / 2.0], dtype=np.float32)
        key = "{}|{}".format(candidate["event_id"], candidate["station"])
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        seed = (
            int(digest[:8], 16) + int(self.cfg.repick_random_seed)
        ) % (2 ** 32)
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, high_rel, size=repeats).astype(np.float32)

    def _flatten_prepared_jobs(self, jobs):
        window_arrays = [job["windows"] for job in jobs]
        window_counts = [len(array) for array in window_arrays]
        windows = np.concatenate(window_arrays, axis=0)
        owners = np.concatenate([
            np.full(count, index, dtype=np.int32)
            for index, count in enumerate(window_counts)
        ])
        starts = np.concatenate([job["window_starts"] for job in jobs])
        for job in jobs:
            job.pop("windows", None)
            job.pop("window_starts", None)
        return windows, owners, starts

    def _run_picker_device_groups(
        self, windows, owners, window_starts, jobs,
    ):
        groups = {}
        for name, picker in self.pickers.items():
            groups.setdefault(str(picker.device), []).append((name, picker))
        results = {}
        picker_seconds = {}
        transfer_seconds = 0.0
        if len(groups) > 1 and self._device_executor is None:
            self._device_executor = ThreadPoolExecutor(
                max_workers=len(groups),
                thread_name_prefix="repick-device",
            )
        if self._device_executor is not None:
            futures = [
                self._device_executor.submit(
                    self._run_picker_device_group,
                    members,
                    windows,
                    owners,
                    window_starts,
                    jobs,
                )
                for members in groups.values()
            ]
            for future in futures:
                group_results, group_seconds, group_transfer = future.result()
                results.update(group_results)
                picker_seconds.update(group_seconds)
                transfer_seconds += group_transfer
        else:
            for members in groups.values():
                group_results, group_seconds, group_transfer = (
                    self._run_picker_device_group(
                        members, windows, owners, window_starts, jobs
                    )
                )
                results.update(group_results)
                picker_seconds.update(group_seconds)
                transfer_seconds += group_transfer
        return results, picker_seconds, transfer_seconds

    def _run_picker_device_group(
        self, members, windows, owners, window_starts, jobs,
    ):
        device = members[0][1].device
        votes = {
            name: [[[], []] for _ in jobs] for name, _ in members
        }
        seconds = {name: 0.0 for name, _ in members}
        transfer_seconds = 0.0
        batch_size = int(self.cfg.repick_batch_size)
        for start in range(0, len(windows), batch_size):
            stop = min(start + batch_size, len(windows))
            transfer_started = time.perf_counter()
            batch = torch.from_numpy(windows[start:stop]).to(
                device, non_blocking=False
            )
            transfer_seconds += time.perf_counter() - transfer_started
            for name, picker in members:
                picker_started = time.perf_counter()
                probabilities = picker.predict_preprocessed_batch(batch)
                for local_index, probability in enumerate(probabilities):
                    flat_index = start + local_index
                    owner = int(owners[flat_index])
                    p_votes, s_votes = picker.picker.decode_one_window(
                        probability, float(window_starts[flat_index])
                    )
                    votes[name][owner][0].extend(p_votes)
                    votes[name][owner][1].extend(s_votes)
                seconds[name] += time.perf_counter() - picker_started
        results = {}
        for name, picker in members:
            results[name] = [
                picker.finalize_shared_votes(
                    job, votes[name][index][0], votes[name][index][1], self.cfg
                )
                for index, job in enumerate(jobs)
            ]
        return results, seconds, transfer_seconds

    def _make_group_pick(self, group_name, members):
        tp_values = [item["tp"] for item in members.values()]
        ts_values = [item["ts"] for item in members.values()]
        p_probs = [item["p_prob"] for item in members.values()]
        s_probs = [item["s_prob"] for item in members.values()]
        return {
            "group": group_name,
            "tp": UTCDateTime(float(median([float(v) for v in tp_values]))),
            "ts": UTCDateTime(float(median([float(v) for v in ts_values]))),
            "p_prob": float(median(p_probs)),
            "s_prob": float(median(s_probs)),
            "tp_std": float(np.std([float(v) for v in tp_values])),
            "ts_std": float(np.std([float(v) for v in ts_values])),
            "p_prob_std": float(np.std(p_probs)),
            "s_prob_std": float(np.std(s_probs)),
            "num_support": len(members),
            "sources": sorted(members),
            "members": members,
        }

    def _cluster_group_pairs(self, per_picker, group_name, tp_pred, ts_pred):
        prefix = group_name + ":"
        records = []
        for key, pairs in per_picker.items():
            if not key.startswith(prefix):
                continue
            model_name = key[len(prefix):]
            for pair in pairs:
                residual = abs(pair["tp"] - tp_pred) / float(self.cfg.tp_dev)
                residual += abs(pair["ts"] - ts_pred) / float(self.cfg.ts_dev)
                records.append((residual, model_name, pair))
        records.sort(key=lambda item: item[0])
        consumed = set()
        outputs = []
        required = int(self.cfg.repick_group_min_picker_support)
        for seed_index, (_, seed_model, seed) in enumerate(records):
            if seed_index in consumed:
                continue
            members = {seed_model: seed}
            member_indices = {seed_index}
            for model_name in sorted({item[1] for item in records} - {seed_model}):
                matches = [
                    (index, item)
                    for index, (_, name, item) in enumerate(records)
                    if index not in consumed and name == model_name
                    and abs(item["tp"] - seed["tp"]) < float(self.cfg.tp_dev)
                    and abs(item["ts"] - seed["ts"]) < float(self.cfg.ts_dev)
                ]
                if matches:
                    index, item = min(matches, key=lambda value: (
                        abs(value[1]["tp"] - seed["tp"])
                        + abs(value[1]["ts"] - seed["ts"])
                    ))
                    members[model_name] = item
                    member_indices.add(index)
            if len(members) < required:
                continue
            consumed.update(member_indices)
            outputs.append(self._make_group_pick(group_name, members))
        return outputs

    def _cluster_repicker_groups(self, per_picker, tp_pred, ts_pred):
        return {
            group_name: self._cluster_group_pairs(
                per_picker, group_name, tp_pred, ts_pred
            )
            for group_name in ("POS_NEG", "POS")
        }

    def _output_repicker_pick(self, job, selected, provenance, groups):
        # Positive-only models are the timing authority when both groups agree.
        timing = selected["POS"] if "POS" in selected else next(
            iter(selected.values())
        )
        tp, ts = timing["tp"], timing["ts"]
        all_members = {}
        for group_name, group_pick in selected.items():
            all_members.update({
                "{}:{}".format(group_name, name): value
                for name, value in group_pick["members"].items()
            })
        vote_ratios = {
            name: float(item["num_votes"]) / float(self.cfg.repick_num_repeat)
            for name, item in all_members.items()
        }
        output = {
            "sta": job["station"],
            "p": tp.datetime,
            "s": ts.datetime,
            # Measure displacement only after reassociation accepts this pick.
            "score": -1.0,
            "p_prob": timing["p_prob"],
            "s_prob": timing["s_prob"],
            "tp_std": timing["tp_std"],
            "ts_std": timing["ts_std"],
            "p_prob_std": timing["p_prob_std"],
            "s_prob_std": timing["s_prob_std"],
            "num_support": sum(item["num_support"] for item in selected.values()),
            "sources": "|".join(sorted(all_members)),
            "picker_window_vote_ratios": format_picker_window_vote_ratios(
                vote_ratios
            ),
            "picker_uncertainties": _format_picker_uncertainties(all_members),
            "pick_provenance": provenance,
        }
        output["quality"] = repick_quality_code(
            provenance, output["picker_window_vote_ratios"], self.cfg
        )
        return output

    def _combine_repicker_groups(self, job, group_results):
        pos_neg = list(group_results["POS_NEG"])
        pos = list(group_results["POS"])
        outputs = []
        used_pos = set()
        for pos_neg_pick in pos_neg:
            matches = [
                (index, pos_pick)
                for index, pos_pick in enumerate(pos)
                if index not in used_pos
                and abs(pos_pick["tp"] - pos_neg_pick["tp"])
                <= float(self.cfg.tp_dev)
                and abs(pos_pick["ts"] - pos_neg_pick["ts"])
                <= float(self.cfg.ts_dev)
            ]
            if matches:
                index, pos_pick = min(matches, key=lambda value: (
                    abs(value[1]["tp"] - pos_neg_pick["tp"])
                    + abs(value[1]["ts"] - pos_neg_pick["ts"])
                ))
                used_pos.add(index)
                outputs.append(self._output_repicker_pick(
                    job, {"POS_NEG": pos_neg_pick, "POS": pos_pick},
                    "both_groups", ["POS_NEG", "POS"],
                ))
            else:
                outputs.append(self._output_repicker_pick(
                    job, {"POS_NEG": pos_neg_pick}, "pos_neg_only", ["POS_NEG"]
                ))
        for index, pos_pick in enumerate(pos):
            if index not in used_pos:
                outputs.append(self._output_repicker_pick(
                    job, {"POS": pos_pick}, "pos_only", ["POS"]
                ))
        return outputs

    def _build_jobs(self, events, realtime_stream_end=None):
        jobs = {}
        num_late_s_events = 0
        stream_end = (
            UTCDateTime(realtime_stream_end)
            if realtime_stream_end is not None else None
        )
        for event_index, event in enumerate(events):
            event_stations = set()
            max_predicted_s = None
            initial_distances = []
            for pick in event["picks"]:
                geometry_key = self._station_geometry_key(pick["sta"])
                if geometry_key is None:
                    continue
                distance = self._epicentral_distance_km(event, geometry_key)
                if distance is not None:
                    initial_distances.append(distance)
            if not initial_distances:
                continue
            adaptive_max_distance = max(initial_distances)
            # Initial station identities define only this geometric radius;
            # their P/S times never enter repicking or reassociation.
            for station, geometry in self.stations.items():
                dx, dy, _ = _distance_components_km(event, geometry)
                epicentral_distance = math.hypot(dx, dy)
                if epicentral_distance <= adaptive_max_distance:
                    event_stations.add(station)

            if stream_end is not None:
                for station in event_stations:
                    dx, dy, dz = _distance_components_km(
                        event, self.stations[station]
                    )
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                    predicted_s = (
                        UTCDateTime(event["time"])
                        + distance / float(self.cfg.vs)
                    )
                    if (
                        max_predicted_s is None
                        or predicted_s > max_predicted_s
                    ):
                        max_predicted_s = predicted_s
            if (
                stream_end is not None
                and max_predicted_s is not None
                and max_predicted_s > stream_end
            ):
                num_late_s_events += 1
                continue
            for station in sorted(event_stations):
                jobs.setdefault(station, []).append(event_index)
        return jobs, num_late_s_events

    def _epicentral_distance_km(self, event, station):
        geometry_key = self._station_geometry_key(station)
        if geometry_key is None:
            return None
        dx, dy, _ = _distance_components_km(
            event, self.stations[geometry_key]
        )
        return math.hypot(dx, dy)

    def _pick_epicentral_distance_km(self, event, pick):
        return self._epicentral_distance_km(event, pick["sta"])

    def _waveform_holder(self, station_waveforms, station):
        if station in station_waveforms:
            return station_waveforms[station]
        base = _station_base(station)
        if base in station_waveforms:
            return station_waveforms[base]
        for key, value in station_waveforms.items():
            if _station_base(key) == base:
                return value
        return None

    def _waveform_entries(self, waveform_context):
        """Normalize offline day maps and realtime segment entries."""
        if isinstance(waveform_context, (list, tuple)):
            return sorted(
                waveform_context,
                key=lambda item: UTCDateTime(item["start"]),
            )
        entries = []
        for observed_date, waveforms in waveform_context.items():
            start = UTCDateTime(observed_date)
            entries.append({
                "start": start,
                "end": start + 86400,
                "waveforms": waveforms,
            })
        return sorted(entries, key=lambda item: UTCDateTime(item["start"]))

    def _merged_station_stream(
        self, waveform_context, station, start_time=None, end_time=None,
    ):
        entries = self._waveform_entries(waveform_context)
        if len(entries) == 1:
            holder = self._waveform_holder(entries[0]["waveforms"], station)
            if holder is not None and holder.stream:
                selector_codes = str(station).split(".")
                channel_family = (
                    selector_codes[2] if len(selector_codes) > 2 else ""
                )
                if (
                    len(holder.stream) == int(self.cfg.num_chn)
                    and (
                        not channel_family
                        or any(
                            trace.stats.channel.startswith(channel_family)
                            for trace in holder.stream
                        )
                    )
                ):
                    # Realtime contexts already own one merged, filtered
                    # station stream. Return it read-only instead of copying
                    # the full segment before slicing short event windows.
                    return holder.stream
        stream = Stream()
        seen = set()
        selector_codes = str(station).split(".")
        channel_family = selector_codes[2] if len(selector_codes) > 2 else ""
        for entry in entries:
            holder = self._waveform_holder(
                entry["waveforms"], station
            )
            if holder is None or not holder.stream or id(holder) in seen:
                continue
            if channel_family and not any(
                trace.stats.channel.startswith(channel_family)
                for trace in holder.stream
            ):
                continue
            seen.add(id(holder))
            segment_start = UTCDateTime(entry["start"])
            segment_end = UTCDateTime(entry["end"])
            sampling_rate = min(
                float(trace.stats.sampling_rate) for trace in holder.stream
            )
            segment_end -= 0.5 / sampling_rate
            if start_time is not None:
                segment_start = max(segment_start, UTCDateTime(start_time))
            if end_time is not None:
                segment_end = min(segment_end, UTCDateTime(end_time))
            if segment_end < segment_start:
                continue
            owned_segment = holder.stream.slice(
                segment_start, segment_end, nearest_sample=True
            )
            for trace in owned_segment:
                stream.append(trace.copy())
        if not stream:
            return None
        try:
            stream.merge(method=1, fill_value=0)
        except Exception as exc:
            print("skip repick waveform {}: {}".format(station, exc), flush=True)
            return None
        if len(stream) != int(self.cfg.num_chn):
            return None
        return stream

    def _predicted_phases(self, event, station):
        dx, dy, dz = _distance_components_km(event, self.stations[station])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        origin_time = UTCDateTime(event["time"])
        return (
            origin_time + distance / float(self.cfg.vp),
            origin_time + distance / float(self.cfg.vs),
        )

    def _reassociate_event(self, event):
        """Associate both-group anchors, then attach matching single-group picks."""
        anchor_picks = []
        supplemental_picks = []
        for pick in event["picks"]:
            station = self._station_geometry_key(pick["sta"])
            if station is None:
                continue
            provenance = resolve_pick_provenance([
                pick.get("pick_provenance", "initial")
            ])
            if provenance in {"both_groups", "repicked"}:
                anchor_picks.append((station, pick))
            elif provenance in {"pos_neg_only", "pos_only"}:
                supplemental_picks.append((station, pick))

        dtype = np.dtype([
            ("net_sta", object),
            ("sta_ot", object),
            ("tp", object),
            ("ts", object),
            ("s_amp", np.float64),
            ("pick_index", np.int32),
        ])
        original_picks = []
        rows = []
        vp = float(getattr(self.cfg, "vp", 6.0))
        vs = float(getattr(self.cfg, "vs", 3.45))
        for station, pick in sorted(
            anchor_picks,
            key=lambda item: (item[0], item[1]["p"], item[1]["s"]),
        ):
            tp = UTCDateTime(pick["p"])
            ts = UTCDateTime(pick["s"])
            if ts <= tp:
                continue
            distance = (ts - tp) / (1.0 / vs - 1.0 / vp)
            sta_ot = tp - distance / vp
            pick_index = len(original_picks)
            original_picks.append(pick)
            rows.append((
                station, sta_ot, tp, ts, float(pick["score"]), pick_index,
            ))
        if len({row[0] for row in rows}) < int(
            self.reassociation_params["min_sta"]
        ):
            return []

        associated = self.reassociator.associate(
            np.asarray(rows, dtype=dtype), verbose=False,
            unique_stations=True,
        )
        if not associated:
            return []
        event_locations, event_pick_sets = associated
        if not event_locations:
            return []

        outputs = []
        for index, location in enumerate(event_locations):
            selected_picks = [
                original_picks[int(row["pick_index"])]
                for row in event_pick_sets[index]
            ]
            candidate = {
                "source": event.get("source", "post-repick-reassociation"),
                "time": UTCDateTime(location["evt_ot"]).datetime,
                "lat": float(location["evt_lat"]),
                "lon": float(location["evt_lon"]),
                "depth": float(location["evt_dep"]),
                "mag": float(location.get("mag", event.get("mag", -1.0))),
                "picks": list(selected_picks),
            }
            anchor_stations = {
                self._station_geometry_key(pick["sta"])
                for pick in selected_picks
            }
            for station, pick in supplemental_picks:
                if station in anchor_stations:
                    continue
                tp_pred, ts_pred = self._predicted_phases(candidate, station)
                if (
                    abs(UTCDateTime(pick["p"]) - tp_pred)
                    <= float(self.cfg.tp_dev)
                    and abs(UTCDateTime(pick["s"]) - ts_pred)
                    <= float(self.cfg.ts_dev)
                ):
                    candidate["picks"].append(pick)
            candidate["picks"].sort(
                key=lambda pick: (pick["sta"], pick["p"], pick["s"])
            )
            outputs.append(candidate)
        outputs.sort(key=lambda candidate: (
            -len({pick["sta"] for pick in candidate["picks"]}),
            candidate["time"],
        ))
        return outputs

    def _measure_final_event_waveform_qc(self, event, amplitude_streams):
        """Measure amplitude, P SNR, and magnitude for a final PAL candidate."""
        dtype = np.dtype([
            ("net_sta", object),
            ("s_amp", np.float64),
        ])
        amplitude_rows = []
        for pick in event["picks"]:
            station = pick["sta"]
            geometry_station = self._station_geometry_key(station)
            stream = amplitude_streams.get(station)
            if stream is None and geometry_station is not None:
                stream = amplitude_streams.get(geometry_station)
            amplitude = -1.0
            p_snr = (-1.0, -1.0, -1.0)
            if stream is not None:
                amplitude = self._displacement_amplitude(
                    stream, UTCDateTime(pick["p"]), UTCDateTime(pick["s"])
                )
                p_snr = self._p_energy_snr(
                    stream, UTCDateTime(pick["p"])
                )
            pick["score"] = amplitude
            pick["p_snr_e"], pick["p_snr_n"], pick["p_snr_z"] = p_snr
            if geometry_station is not None:
                amplitude_rows.append((geometry_station, amplitude))

        event_location = {
            "evt_ot": UTCDateTime(event["time"]),
            "evt_lat": float(event["lat"]),
            "evt_lon": float(event["lon"]),
            "evt_dep": float(event["depth"]),
            "mag": -1.0,
        }
        if amplitude_rows:
            event_location = self.reassociator.calc_mag(
                np.asarray(amplitude_rows, dtype=dtype), event_location
            )
        event["mag"] = float(event_location.get("mag", -1.0))

    @staticmethod
    def _pal_energy_sta_lta(data, win_lta_npts, win_sta_npts):
        """Return PAL's forward-STA/backward-LTA ratio for energy data."""
        data = np.asarray(data, dtype=np.float64)
        npts = len(data)
        if npts < win_lta_npts + win_sta_npts:
            return np.zeros(npts, dtype=np.float64)
        sta = np.zeros(npts, dtype=np.float64)
        lta = np.ones(npts, dtype=np.float64)
        data_cum = np.cumsum(data)
        sta[:-win_sta_npts] = (
            data_cum[win_sta_npts:] - data_cum[:-win_sta_npts]
        )
        sta /= win_sta_npts
        lta[win_lta_npts:] = (
            data_cum[win_lta_npts:] - data_cum[:-win_lta_npts]
        )
        lta /= win_lta_npts
        with np.errstate(divide="ignore", invalid="ignore"):
            sta_lta = sta / lta
        sta_lta[:win_lta_npts] = 0.0
        sta_lta[~np.isfinite(sta_lta)] = 0.0
        return sta_lta

    def _p_energy_snr(self, stream, tp):
        """Measure PAL energy STA/LTA around P for normalized E/N/Z traces."""
        tp_dev = float(self.cfg.tp_dev)
        search_start = tp - tp_dev
        search_end = tp + tp_dev
        required_start = search_start - PAL_P_SNR_LTA_SEC
        required_end = search_end + PAL_P_SNR_STA_SEC
        values = {}
        for trace in stream:
            component = str(trace.stats.channel or "")[-1:].upper()
            component = {"1": "E", "2": "N"}.get(component, component)
            if component not in ("E", "N", "Z") or component in values:
                component = next(
                    (name for name in "ENZ" if name not in values), None
                )
            if component is None:
                continue
            sampling_rate = float(trace.stats.sampling_rate)
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                values[component] = -1.0
                continue
            tolerance = 0.51 / sampling_rate
            if (
                trace.stats.starttime > required_start + tolerance
                or trace.stats.endtime < required_end - tolerance
            ):
                values[component] = -1.0
                continue
            window = trace.slice(
                required_start, required_end, nearest_sample=True
            )
            data = np.asarray(window.data, dtype=np.float64)
            if len(data) < 2 or not np.all(np.isfinite(data)):
                values[component] = -1.0
                continue
            win_sta_npts = max(1, int(sampling_rate * PAL_P_SNR_STA_SEC))
            win_lta_npts = max(1, int(sampling_rate * PAL_P_SNR_LTA_SEC))
            characteristic = self._pal_energy_sta_lta(
                data ** 2, win_lta_npts, win_sta_npts
            )
            index_start = int(round(
                float(search_start - window.stats.starttime) * sampling_rate
            ))
            index_end = int(round(
                float(search_end - window.stats.starttime) * sampling_rate
            )) + 1
            search = characteristic[
                max(0, index_start):min(len(characteristic), index_end)
            ]
            snr = float(np.max(search)) if len(search) else -1.0
            values[component] = (
                snr if np.isfinite(snr) and snr >= 0 else -1.0
            )
        return tuple(values.get(component, -1.0) for component in "ENZ")

    def _displacement_amplitude(self, stream, tp, ts):
        return displacement_amplitude(
            stream, tp, ts, self.cfg.amp_win, self.cfg.num_chn
        )

    def _write_outputs(self, events, phase_path, catalog_path):
        phase_path = Path(phase_path)
        catalog_path = Path(catalog_path) if catalog_path else None
        phase_partial = phase_path.with_suffix(phase_path.suffix + ".repick")
        catalog_partial = (
            catalog_path.with_suffix(catalog_path.suffix + ".repick")
            if catalog_path is not None else None
        )
        phase_fp = phase_partial.open("w", encoding="utf-8")
        catalog_fp = (
            catalog_partial.open("w", encoding="utf-8")
            if catalog_partial is not None else None
        )
        try:
            for event in sorted(events, key=lambda item: item["time"]):
                header = "{},{:.5f},{:.5f},{:.1f},{:.2f}\n".format(
                    format_time(event["time"], self.cfg.merge_time_format_digits),
                    event["lat"],
                    event["lon"],
                    event["depth"],
                    event["mag"],
                )
                phase_fp.write(header)
                if catalog_fp is not None:
                    catalog_fp.write(header)
                for pick in sorted(event["picks"], key=lambda item: item["sta"]):
                    phase_fp.write(
                        "{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},"
                        "{:.4f},{:.4f},{},{},{},{},{},"
                        "{:.4f},{:.4f},{:.4f}\n".format(
                            pick["sta"],
                            format_time(
                                pick["p"], self.cfg.merge_time_format_digits
                            ),
                            format_time(
                                pick["s"], self.cfg.merge_time_format_digits
                            ),
                            pick["score"],
                            pick.get("quality", -1),
                            pick["p_prob"],
                            pick["s_prob"],
                            pick["tp_std"],
                            pick["ts_std"],
                            pick["p_prob_std"],
                            pick["s_prob_std"],
                            pick["num_support"],
                            pick["sources"],
                            pick.get("picker_window_vote_ratios", ""),
                            pick.get("picker_uncertainties", ""),
                            pick.get("pick_provenance", "initial"),
                            pick.get("p_snr_e", -1.0),
                            pick.get("p_snr_n", -1.0),
                            pick.get("p_snr_z", -1.0),
                        )
                    )
        finally:
            phase_fp.close()
            if catalog_fp is not None:
                catalog_fp.close()
        phase_partial.replace(phase_path)
        if catalog_partial is not None:
            catalog_partial.replace(catalog_path)


class RealtimeEventRepickCoordinator(object):
    """Repick merged segment detections before segment finalization."""

    STATUS_FIELDS = (
        "time", "repick_version", "picker_group", "branch", "interval_start",
        "interval_end", "phase_path",
        "waveform_segments", "num_events", "num_station_event_attempts",
        "num_repicker_phase_pairs_generated",
        "num_picks_reassociation_rejected",
        "num_events_reassociated", "num_events_reassociation_rejected",
        "num_late_s_events_skipped", "num_repick_windows",
        "job_build_sec", "window_prepare_sec", "device_inference_wall_sec",
        "device_transfer_sec", "result_merge_sec", "reassociation_sec",
        "waveform_qc_measurement_sec",
        "output_write_sec",
        "picker_time_sec", "elapsed_sec",
    )

    def __init__(self, repicker, status_path, max_cached_segments=1):
        self.repicker = repicker
        self.status_path = Path(status_path)
        self.max_cached_segments = max(1, int(max_cached_segments))
        self.picker_group = "|".join(
            "{}:{}".format(group_name, picker_name)
            for group_name in sorted(repicker.repicker_specs)
            for picker_name in sorted(repicker.repicker_specs[group_name])
        )
        self.segment_cache = {}
        self.final_waveform_snapshots = {"AI-PAL": []}
        for branch_name in getattr(
            repicker.cfg, "event_waveform_plot_ref_branches", []
        ):
            self.final_waveform_snapshots[branch_name] = []
        self.snapshot_cache_root = os.path.join(
            repicker.cfg.out_root, "_internal", "event_waveform_cache"
        )
        # Deferred snapshots cannot survive a process restart because their
        # Python event metadata is intentionally not persisted.
        if os.path.isdir(self.snapshot_cache_root):
            shutil.rmtree(self.snapshot_cache_root, ignore_errors=True)
        self._status_needs_reset = False
        self.completed_phase_paths = self._load_completed_phase_paths()

    def begin_segment(self, segment):
        """Release prior waveform segments before preparing a new one."""
        self.release_segment_cache("before {}".format(segment))

    def release_segment_cache(self, reason=""):
        """Release filtered station waveforms once segment repicking is done."""
        removed = sorted(self.segment_cache)
        for cached in self.segment_cache.values():
            waveforms = cached.get("waveforms", {})
            for holder in waveforms.values():
                release = getattr(holder, "release", None)
                if release is not None:
                    release()
            waveforms.clear()
        self.segment_cache.clear()
        if removed:
            suffix = " {}".format(reason) if reason else ""
            print(
                "released realtime repick waveform cache{}: {}".format(
                    suffix, removed
                ),
                flush=True,
            )

    def close(self):
        """Release active segments and deferred plot snapshots."""
        self.release_segment_cache("during shutdown")
        for snapshots in self.final_waveform_snapshots.values():
            for snapshot in snapshots:
                self._release_snapshot(snapshot)
        self.final_waveform_snapshots.clear()
        shutil.rmtree(self.snapshot_cache_root, ignore_errors=True)
        self.repicker.close()
        _trim_cpu_allocator()

    def _load_completed_phase_paths(self):
        if not self.status_path.exists():
            return set()
        with self.status_path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if tuple(reader.fieldnames or ()) != self.STATUS_FIELDS:
                self._status_needs_reset = True
                return set()
            rows = list(reader)
            if any(
                row.get("picker_group") != self.picker_group for row in rows
            ):
                self._status_needs_reset = True
                return set()
            completed = set()
            for row in rows:
                if (
                    not row.get("phase_path")
                    or row.get("repick_version") != EVENT_REPICK_VERSION
                ):
                    continue
                phase_path = os.path.abspath(row["phase_path"])
                if not os.path.exists(phase_path):
                    continue
                events = read_phase_file(phase_path)
                # A zero-event file can be a valid QC result. Nonempty files
                # must contain only picks produced by the two repicker groups;
                # otherwise a later initial-association rewrite made the old
                # status row stale.
                if all(
                    pick.get("pick_provenance") in {
                        "both_groups", "pos_neg_only", "pos_only",
                    }
                    for event in events for pick in event.get("picks", [])
                ):
                    completed.add(phase_path)
                else:
                    print(
                        "stale repick status ignored for initial-content "
                        "phase file {}".format(phase_path),
                        flush=True,
                    )
            return completed

    def register_segment(self, segment, start, end, waveforms):
        start = UTCDateTime(start)
        end = UTCDateTime(end)
        if end <= start:
            raise ValueError("invalid realtime waveform segment bounds")
        self.segment_cache[segment] = {
            "segment": segment,
            "start": start,
            "end": end,
            "waveforms": waveforms,
        }
        ordered = sorted(
            self.segment_cache.values(), key=lambda item: item["end"]
        )
        while len(ordered) > self.max_cached_segments:
            removed = ordered.pop(0)
            self.segment_cache.pop(removed["segment"], None)
            for holder in removed.get("waveforms", {}).values():
                release = getattr(holder, "release", None)
                if release is not None:
                    release()
            removed.get("waveforms", {}).clear()
        print(
            "realtime repick waveform cache: {}".format(
                [item["segment"] for item in ordered]
            ),
            flush=True,
        )

    def process_segment_result(
        self, branch_name, segment, phase_path, catalog_path="", force=False,
    ):
        phase_record = os.path.abspath(phase_path)
        if force:
            self.completed_phase_paths.discard(phase_record)
        if phase_record in self.completed_phase_paths:
            return None
        selected = self.segment_cache.get(segment)
        if selected is None:
            print(
                "warning: realtime event repick skipped {}: missing "
                "filtered segment {}".format(phase_path, segment),
                flush=True,
            )
            return None
        repick_context = {
            "repick_stream_end": (
                selected["end"]
                - float(self.repicker.cfg.taper_max_length_sec)
            ),
            "segment_local_waveforms": True,
            "defer_event_waveform_plot": bool(getattr(
                self.repicker.cfg, "enable_event_waveform_plot", False
            )),
        }
        summary = self.repicker.process_hour(
            selected["start"],
            selected["end"],
            phase_path,
            catalog_path,
            [selected],
            repick_context,
        )
        self.final_waveform_snapshots["AI-PAL"].extend(
            summary.pop("_event_waveform_snapshots", [])
        )
        summary.update({
            "branch": branch_name,
            "interval_start": str(selected["start"]),
            "interval_end": str(selected["end"]),
            "phase_path": phase_path,
            "waveform_segments": segment,
            "picker_time_sec": "|".join(
                "{}:{:.6f}".format(name, seconds)
                for name, seconds in sorted(
                    summary.get("picker_seconds", {}).items()
                )
            ),
        })
        self._append_status(summary)
        self.completed_phase_paths.add(phase_record)
        return summary

    def capture_reference_result(self, branch_name, segment, phase_path):
        """Retain filtered windows for one selected reference branch."""
        if branch_name not in self.final_waveform_snapshots:
            return 0
        selected = self.segment_cache.get(segment)
        if selected is None:
            print(
                "warning: reference event waveform capture skipped {}: "
                "missing filtered segment {}".format(branch_name, segment),
                flush=True,
            )
            return 0
        events = read_phase_file(phase_path)
        snapshots = self.repicker.capture_event_waveforms(events, [selected])
        self.final_waveform_snapshots[branch_name].extend(snapshots)
        print(
            "reference event waveform snapshots: {} | {} events".format(
                branch_name, len(snapshots)
            ),
            flush=True,
        )
        return len(snapshots)

    def _snapshot_matches_event(self, snapshot, event):
        groups = group_events(
            [snapshot["event"], event],
            self.repicker.cfg.merge_origin_time_tol_sec,
            self.repicker.cfg.merge_epicenter_tol_km,
            self.repicker.cfg.merge_depth_tol_km,
            self.repicker.cfg.merge_min_shared_phase_stations,
            self.repicker.cfg.merge_phase_pick_time_tol_sec,
        )
        return len(groups) == 1

    @staticmethod
    def _release_snapshot(snapshot):
        for holder in snapshot.get("waveforms", {}).values():
            release = getattr(holder, "release", None)
            if release is not None:
                release()
        snapshot.get("waveforms", {}).clear()

    def plot_final_results(self, final_results, branch_name="AI-PAL"):
        """Plot published final events, then release finalized segment data."""
        is_preferred = branch_name == "AI-PAL"
        enabled = (
            bool(getattr(
                self.repicker.cfg, "enable_event_waveform_plot", False
            ))
            if is_preferred else branch_name in getattr(
                self.repicker.cfg, "event_waveform_plot_ref_branches", []
            )
        )
        if not enabled:
            return {
                "num_final_events": 0,
                "num_plots": 0,
                "num_missing_snapshots": 0,
                "num_unrendered_waveforms": 0,
                "plot_sec": 0.0,
                "released": 0,
                "pending": len(self.final_waveform_snapshots.get(
                    branch_name, []
                )),
            }
        started = time.perf_counter()
        num_final_events = 0
        num_plots = 0
        num_missing_snapshots = 0
        num_unrendered_waveforms = 0
        released = 0
        snapshots = self.final_waveform_snapshots.setdefault(branch_name, [])
        output_dir = Path(
            self.repicker.cfg.out_event_waveform_final_dir
            if is_preferred
            else self.repicker.cfg.out_event_waveform_final_ref_dirs[branch_name]
        )
        for result in sorted(
            final_results, key=lambda item: item["interval_start"]
        ):
            final_events = read_phase_file(result["phase_path"])
            for event in final_events:
                num_final_events += 1
                matched = [
                    snapshot for snapshot in snapshots
                    if self._snapshot_matches_event(snapshot, event)
                ]
                if not matched:
                    print(
                        "warning: no retained event waveform matches final "
                        "event {}".format(format_time(event["time"])),
                        flush=True,
                    )
                    num_missing_snapshots += 1
                    continue
                plotted = self.repicker._plot_events(
                    [event], matched, output_dir
                )
                num_plots += plotted
                if plotted == 0:
                    num_unrendered_waveforms += 1
                    print(
                        "warning: retained snapshot for final event {} has "
                        "no usable waveform rows".format(
                            format_time(event["time"])
                        ),
                        flush=True,
                    )
                # One source snapshot belongs to only one finalized event.
                # Release its lazily loaded arrays immediately after plotting.
                matched_ids = {id(snapshot) for snapshot in matched}
                for snapshot in matched:
                    self._release_snapshot(snapshot)
                    released += 1
                snapshots = [
                    snapshot for snapshot in snapshots
                    if id(snapshot) not in matched_ids
                ]

            interval_end = UTCDateTime(result["interval_end"])
            remaining = []
            for snapshot in snapshots:
                if UTCDateTime(snapshot["event"]["time"]) < interval_end:
                    self._release_snapshot(snapshot)
                    released += 1
                else:
                    remaining.append(snapshot)
            snapshots = remaining
            _trim_cpu_allocator()
        self.final_waveform_snapshots[branch_name] = snapshots
        elapsed = time.perf_counter() - started
        if final_results:
            print(
                "final event waveform plots for {}: {} published events | "
                "{} PNG | {} missing snapshots | {} unusable waveforms | "
                "{} snapshots released | {} snapshots pending | {:.2f}s".format(
                    branch_name, num_final_events, num_plots,
                    num_missing_snapshots, num_unrendered_waveforms, released,
                    len(snapshots),
                    elapsed,
                ),
                flush=True,
            )
        return {
            "num_final_events": num_final_events,
            "num_plots": num_plots,
            "num_missing_snapshots": num_missing_snapshots,
            "num_unrendered_waveforms": num_unrendered_waveforms,
            "plot_sec": elapsed,
            "released": released,
            "pending": len(snapshots),
        }

    def _append_status(self, summary):
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if self._status_needs_reset and self.status_path.exists():
            self.status_path.unlink()
            self._status_needs_reset = False
        write_header = not self.status_path.exists()
        with self.status_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.STATUS_FIELDS)
            if write_header:
                writer.writeheader()
            row = {name: summary.get(name, "") for name in self.STATUS_FIELDS}
            row["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            row["repick_version"] = EVENT_REPICK_VERSION
            row["picker_group"] = self.picker_group
            writer.writerow(row)
            fp.flush()
            os.fsync(fp.fileno())
