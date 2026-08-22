"""Pseudo-realtime data pipeline for multi-model picking and PAL association.

It adapts the shared waveform ingest, native/reference picker inference, picker
consensus, and PAL output layers for all-network miniSEED files.
"""
import csv
import ctypes
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from statistics import median
from threading import Lock

import numpy as np
import torch
from obspy import read, Stream, UTCDateTime

from data_pipeline import select_gain_for_time
from phase_merge import (
    _cluster_station_picks, event_both_group_pick_ratio,
    select_preferred_provenance_picks, write_phase_file,
)
from picker_stream import PreparedPickerStream
from waveform_qc import displacement_amplitude, is_glitch


PREFERRED_ENSEMBLE_BRANCH = "AI-PAL"
from pick_ensemble import (
    format_pick_row, format_picker_cluster_sizes, merge_picker_cluster_sizes,
    merge_picker_records,
)


PICK_DTYPE = [
    ("net_sta", "O"),
    ("sta_ot", "O"),
    ("tp", "O"),
    ("ts", "O"),
    ("s_amp", "O"),
    ("p_prob", "O"),
    ("s_prob", "O"),
    ("tp_std", "O"),
    ("ts_std", "O"),
    ("p_prob_std", "O"),
    ("s_prob_std", "O"),
    ("num_support", "O"),
    ("sources", "O"),
    ("picker_cluster_sizes", "O"),
]

ENSEMBLE_PICK_DTYPE = PICK_DTYPE


def _station_base(selector):
    return ".".join(str(selector).split(".")[:2])


def _matching_mapping_value(mapping, station):
    if station in mapping:
        return mapping[station]
    base = _station_base(station)
    if base in mapping:
        return mapping[base]
    for key, value in mapping.items():
        if _station_base(key) == base:
            return value
    return None


def _initial_event_magnitude(event, station_dict):
    magnitudes = []
    for pick in event["picks"]:
        geometry = _matching_mapping_value(station_dict, pick["sta"])
        amplitude = float(pick.get("score", -1.0))
        if geometry is None or not np.isfinite(amplitude) or amplitude <= 0:
            continue
        sta_lat, sta_lon, sta_ele = [float(value) for value in geometry[:3]]
        dist_lat = 111.0 * (sta_lat - float(event["lat"]))
        dist_lon = (
            111.0 * (sta_lon - float(event["lon"]))
            * np.cos(sta_lat * np.pi / 180.0)
        )
        dist_dep = float(event["depth"]) + sta_ele / 1000.0
        distance = np.sqrt(
            dist_lon ** 2 + dist_lat ** 2 + dist_dep ** 2
        )
        if np.isfinite(distance) and distance > 0:
            magnitudes.append(
                np.log10(amplitude * 1e6) + np.log10(distance) + 1.0
            )
    if len(magnitudes) >= 3:
        values = np.asarray(magnitudes, dtype=float)
        values = np.delete(values, np.argmax(abs(values - np.median(values))))
        magnitudes = values.tolist()
    event["mag"] = (
        round(float(np.median(magnitudes)), 2) if magnitudes else -1.0
    )


def qc_initial_phase_file(
    phase_path, retained_waveforms, station_dict, cfg, cache=None,
):
    """Measure associated picks only, reject glitches, and refresh magnitude."""
    events = read_phase_file(phase_path)
    cache = {} if cache is None else cache
    by_station = {}
    for event in events:
        for pick in event["picks"]:
            by_station.setdefault(pick["sta"], []).append(pick)

    rejected_picks = 0
    for station, picks in by_station.items():
        holder = _matching_mapping_value(retained_waveforms, station)
        stream = holder.stream if holder is not None else None
        try:
            for pick in picks:
                key = (
                    str(station),
                    round(pick["p"].timestamp(), 4),
                    round(pick["s"].timestamp(), 4),
                )
                result = cache.get(key)
                if result is None:
                    if stream is None:
                        result = (False, -1.0)
                    else:
                        glitch = bool(getattr(cfg, "rm_glitch", True)) and (
                            is_glitch(
                                stream, UTCDateTime(pick["p"]),
                                UTCDateTime(pick["s"]), cfg,
                            )
                        )
                        amplitude = (
                            -1.0 if glitch else displacement_amplitude(
                                stream,
                                UTCDateTime(pick["p"]),
                                UTCDateTime(pick["s"]),
                                cfg.amp_win,
                                cfg.num_chn,
                            )
                        )
                        result = (glitch, amplitude)
                    cache[key] = result
                pick["_initial_glitch"] = result[0]
                pick["score"] = result[1]
                rejected_picks += int(result[0])
        finally:
            if holder is not None and hasattr(holder, "unload"):
                holder.unload()

    params = dict(getattr(cfg, "subnet_assoc_params", {}).get("default", {}))
    params.update(getattr(cfg, "subnet_assoc_params", {}).get("full", {}))
    min_sta = int(params.get("min_sta", 4))
    accepted = []
    for event in events:
        event["picks"] = [
            pick for pick in event["picks"]
            if not pick.pop("_initial_glitch", False)
        ]
        if len({_station_base(pick["sta"]) for pick in event["picks"]}) < min_sta:
            continue
        _initial_event_magnitude(event, station_dict)
        accepted.append(event)
    write_phase_file(
        phase_path, accepted,
        time_format_digits=int(cfg.merge_time_format_digits),
    )
    return {
        "num_input_events": len(events),
        "num_output_events": len(accepted),
        "num_rejected_events": len(events) - len(accepted),
        "num_rejected_picks": rejected_picks,
    }


class BadRealtimeInputError(RuntimeError):
    """A MiniSEED file cannot be decoded or normalized into a valid stream."""


def process_rss_mb():
    """Return current Linux resident memory, or NaN when unavailable."""
    try:
        with open("/proc/self/status") as fp:
            for line in fp:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return float("nan")


def process_rss_breakdown_mb():
    """Return Linux anonymous, file-backed, and shared resident memory."""
    values = {
        "RssAnon:": float("nan"),
        "RssFile:": float("nan"),
        "RssShmem:": float("nan"),
    }
    try:
        with open("/proc/self/status") as fp:
            for line in fp:
                key = line.split(None, 1)[0] if line.strip() else ""
                if key in values:
                    values[key] = float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return values["RssAnon:"], values["RssFile:"], values["RssShmem:"]


def cgroup_memory_mb():
    """Return total Linux cgroup usage and limit for this pipeline."""
    host_limit_mb = float("nan")
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                if line.startswith("MemTotal:"):
                    host_limit_mb = float(line.split()[1]) / 1024.0
                    break
    except (OSError, ValueError, IndexError):
        pass
    candidates = (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    )
    for usage_path, limit_path in candidates:
        try:
            with open(usage_path) as fp:
                usage_text = fp.read().strip()
            with open(limit_path) as fp:
                limit_text = fp.read().strip()
            usage = float(usage_text) / 1024.0 ** 2
            if limit_text == "max":
                limit = host_limit_mb
            else:
                limit = float(limit_text) / 1024.0 ** 2
                # cgroup v1 represents unlimited memory with a value near
                # 2**63 bytes. Report physical RAM as the effective ceiling.
                if (
                    np.isfinite(host_limit_mb)
                    and limit > host_limit_mb * 100.0
                ):
                    limit = host_limit_mb
            return usage, limit
        except (OSError, ValueError):
            continue
    return float("nan"), float("nan")


def write_segment_memory_stage(out_dir, segment, stage):
    """Persist stage memory immediately so SIGKILL leaves its last boundary."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "memory_stages_{}.csv".format(segment))
    write_header = not os.path.exists(path)
    extended_schema = write_header
    if not write_header:
        try:
            with open(path, newline="") as existing_fp:
                extended_schema = "rss_anon_mb" in next(csv.reader(existing_fp))
        except (OSError, StopIteration):
            extended_schema = False
    cgroup_mb, cgroup_limit_mb = cgroup_memory_mb()
    rss_anon_mb, rss_file_mb, rss_shmem_mb = process_rss_breakdown_mb()
    cuda_allocated_mb = 0.0
    cuda_reserved_mb = 0.0
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            cuda_allocated_mb += torch.cuda.memory_allocated(device_index)
            cuda_reserved_mb += torch.cuda.memory_reserved(device_index)
        cuda_allocated_mb /= 1024.0 ** 2
        cuda_reserved_mb /= 1024.0 ** 2
    with open(path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow([
                "time_utc", "stage", "rss_mb", "cgroup_mb",
                "cgroup_limit_mb", "rss_anon_mb", "rss_file_mb",
                "rss_shmem_mb", "cuda_allocated_mb", "cuda_reserved_mb",
            ])
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), stage,
            round(process_rss_mb(), 2), round(cgroup_mb, 2),
            round(cgroup_limit_mb, 2),
        ]
        if extended_schema:
            row.extend([
                round(rss_anon_mb, 2), round(rss_file_mb, 2),
                round(rss_shmem_mb, 2),
            ])
        row.extend([round(cuda_allocated_mb, 2), round(cuda_reserved_mb, 2)])
        writer.writerow(row)
        fp.flush()
        os.fsync(fp.fileno())


def release_transient_memory():
    """Return completed-segment allocations to CUDA and the Linux allocator."""
    trim_cpu_allocator()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return process_rss_mb()


def trim_cpu_allocator():
    """Return released NumPy/ObsPy arrays to the Linux allocator."""
    gc.collect()
    try:
        # NumPy/ObsPy arrays can remain in glibc arenas after Python releases
        # them, which matters for a long-lived process under a memory limit.
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, AttributeError):
        pass


def write_memory_progress(out_dir, segment, completed, total, rss_mb):
    """Persist station-level memory progress so SIGKILL leaves diagnostics."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "memory_progress_{}.csv".format(segment))
    write_header = completed == 0 or not os.path.exists(path)
    mode = "w" if completed == 0 else "a"
    cgroup_mb, cgroup_limit_mb = cgroup_memory_mb()
    cuda_allocated_mb = 0.0
    cuda_reserved_mb = 0.0
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            cuda_allocated_mb += torch.cuda.memory_allocated(device_index)
            cuda_reserved_mb += torch.cuda.memory_reserved(device_index)
        cuda_allocated_mb /= 1024.0 ** 2
        cuda_reserved_mb /= 1024.0 ** 2
    with open(path, mode, newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow([
                "time_utc", "stations_completed", "stations_total", "rss_mb",
                "cgroup_mb", "cgroup_limit_mb", "cuda_allocated_mb",
                "cuda_reserved_mb",
            ])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            int(completed), int(total), round(float(rss_mb), 2),
            round(cgroup_mb, 2), round(cgroup_limit_mb, 2),
            round(cuda_allocated_mb, 2), round(cuda_reserved_mb, 2),
        ])
        fp.flush()
        os.fsync(fp.fileno())

def _picker_device_groups(pickers):
    groups = {}
    for picker_name, picker in pickers.items():
        groups.setdefault(str(picker.device), []).append((picker_name, picker))
    return groups


class RealtimeInferenceExecutors(object):
    """Reusable CPU preparation and device-dispatch worker pools."""

    def __init__(self, pickers, num_workers):
        self.num_workers = int(num_workers or 0)
        self.num_devices = len(_picker_device_groups(pickers))
        self.station = None
        self.picker = None
        self._start()

    def _start(self):
        self.station = (
            ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="station-preprocess",
            )
            if self.num_workers > 1 else None
        )
        self.picker = (
            ThreadPoolExecutor(
                max_workers=self.num_devices,
                thread_name_prefix="picker-device",
            )
            if self.num_devices > 1 else None
        )

    def _shutdown(self):
        if self.station is not None:
            self.station.shutdown(wait=True, cancel_futures=True)
            self.station = None
        if self.picker is not None:
            self.picker.shutdown(wait=True, cancel_futures=True)
            self.picker = None

    def recycle(self):
        """Release per-thread allocator arenas between realtime segments."""
        self._shutdown()
        trim_cpu_allocator()
        self._start()

    def close(self):
        self._shutdown()
        trim_cpu_allocator()


def _run_picker_device_group(members, prepared, device_lock):
    results = {}
    seconds = {}
    with device_lock:
        # The first native picker assigned to this device creates the tensor;
        # subsequent native pickers reuse the cached base tensor.
        for picker_name, picker in members:
            t0 = time.perf_counter()
            with torch.inference_mode():
                picks = picker.pick(prepared.stream, prepared=prepared)
            results[picker_name] = picks if picks else []
            seconds[picker_name] = time.perf_counter() - t0
    return results, seconds


def prepare_and_pick_station(
    pickers, st_all, sta_key, sta_info, location_priority, preprocess_cfg,
    device_groups, device_locks, picker_executor=None,
):
    """Prepare one station once, then run device-grouped picker inference."""
    t0 = time.perf_counter()
    prepared = prepare_station_waveform(
        st_all, sta_key, sta_info, location_priority, preprocess_cfg
    )
    if prepared is None:
        return sta_key, {}, time.perf_counter() - t0, {}, False, None
    preprocess_sec = time.perf_counter() - t0

    print("-" * 40)
    print("picking {} with shared preprocessing: {}".format(
        sta_key, ", ".join(pickers)
    ))
    picker_results = {}
    picker_seconds = {}
    if picker_executor is not None:
        futures = [
            picker_executor.submit(
                _run_picker_device_group,
                members,
                prepared,
                device_locks[device],
            )
            for device, members in device_groups.items()
        ]
        for future in futures:
            results, seconds = future.result()
            picker_results.update(results)
            picker_seconds.update(seconds)
    else:
        for device, members in device_groups.items():
            results, seconds = _run_picker_device_group(
                members, prepared, device_locks[device]
            )
            picker_results.update(results)
            picker_seconds.update(seconds)
    retained = prepared.retained_waveform()
    return (
        sta_key, picker_results, preprocess_sec, picker_seconds, True, retained
    )


def _bounded_station_results(
    executor, station_keys, max_pending, worker,
):
    """Yield station results as completed with a bounded future population."""
    station_iter = iter(station_keys)
    pending = set()

    def submit_next():
        try:
            station_key = next(station_iter)
        except StopIteration:
            return False
        pending.add(executor.submit(worker, station_key))
        return True

    for _ in range(max(1, int(max_pending))):
        if not submit_next():
            break
    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()
            submit_next()


def prepare_station_waveform(
    st_all, sta_key, sta_info, location_priority, preprocess_cfg,
):
    """Prepare one station without invoking a picker."""
    st = traces_for_station(st_all, sta_key, location_priority)
    if len(st) != 0:
        st = normalize_station_stream(st, sta_key, sta_info)
    if len(st) != 3:
        return None
    return PreparedPickerStream.from_raw_stream(st, preprocess_cfg)


def prepare_realtime_segment_waveforms(mseed_path, sta_dict, cfg):
    """Read and preprocess one historical segment for startup repicking."""
    timing = {}
    segment = segment_code(mseed_path)
    waveform_spill_dir = os.path.join(
        cfg.out_root, "_internal", "waveform_cache", segment
    )
    shutil.rmtree(waveform_spill_dir, ignore_errors=True)
    os.makedirs(waveform_spill_dir, exist_ok=True)
    st_all = read_realtime_mseed_timed(
        mseed_path, timing, expected_sampling_rate=cfg.samp_rate
    )
    timing["rss_after_mseed_mb"] = process_rss_mb()
    station_keys = sorted(sta_dict)

    def prepare(sta_key):
        prepared = prepare_station_waveform(
            st_all,
            sta_key,
            sta_dict[sta_key],
            cfg.location_priority,
            cfg,
        )
        if prepared is None:
            return sta_key, None
        retained = prepared.retained_waveform()
        spill_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sta_key) + ".pkl"
        retained.spill(os.path.join(waveform_spill_dir, spill_name))
        return sta_key, retained

    waveforms = {}
    started = time.perf_counter()
    if cfg.num_workers and cfg.num_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            results = _bounded_station_results(
                executor, station_keys, cfg.num_workers, prepare
            )
            for sta_key, waveform in results:
                if waveform is not None:
                    waveforms[sta_key] = waveform
                if len(waveforms) % 10 == 0:
                    trim_cpu_allocator()
    else:
        for sta_key in station_keys:
            _, waveform = prepare(sta_key)
            if waveform is not None:
                waveforms[sta_key] = waveform
    timing["preprocess_sec"] = time.perf_counter() - started
    timing["num_station_streams"] = len(waveforms)
    timing["rss_after_picking_mb"] = process_rss_mb()
    del st_all
    gc.collect()
    timing["rss_after_mseed_release_mb"] = process_rss_mb()
    return waveforms, timing

def get_realtime_sta_dict(fsta):
    """Read station metadata keyed by NET.STA.CH_PREFIX.

    The first field is expected to be like ``CI.WWB.HH`` or ``CI.WWB.HN``.
    Gain fields follow the PAL station format and are used to convert counts
    to velocity before either picker runs and before amplitude measurement.
    """
    sta_dict = {}
    with open(fsta) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            codes = [code.strip() for code in line.split(",")]
            if len(codes) < 4:
                print("bad station line: {}".format(line))
                continue
            sta_key = codes[0]
            lat, lon, ele = [float(code) for code in codes[1:4]]
            gain_codes = codes[4:]
            gain = 1.0
            if len(gain_codes) == 1:
                gain = float(gain_codes[0])
            elif len(gain_codes) == 3:
                gain = [float(code) for code in gain_codes]
            elif len(gain_codes) == 5:
                gain = [[float(code) for code in gain_codes[0:3]] + gain_codes[3:5]]
            elif len(gain_codes) > 5 and len(gain_codes) % 5 == 0:
                gain = []
                for ii in range(0, len(gain_codes), 5):
                    gain.append([float(code) for code in gain_codes[ii:ii+3]] + gain_codes[ii+3:ii+5])
            sta_dict[sta_key] = [lat, lon, ele, gain]
    return sta_dict


def parse_sta_key(sta_key):
    codes = sta_key.split(".")
    if len(codes) < 2:
        raise ValueError("station key must start with NET.STA: {}".format(sta_key))
    net, sta = codes[0], codes[1]
    ch_prefix = codes[2] if len(codes) > 2 else ""
    return net, sta, ch_prefix


def segment_code(mseed_path):
    stem = os.path.splitext(os.path.basename(mseed_path))[0]
    return stem


def calc_ot(tp, ts, vp=6.0, vs=3.45):
    dist = (ts - tp) / (1 / vs - 1 / vp)
    return tp - dist / vp

def unpack_picker_pick(pick, picker_name):
    """Normalize native, structured, or legacy picker output."""
    if isinstance(pick, dict):
        value = lambda name, default=None: pick.get(name, default)
    elif hasattr(pick, "dtype") and getattr(pick.dtype, "names", None):
        names = pick.dtype.names
        value = lambda name, default=None: pick[name] if name in names else default
    else:
        legacy = list(pick)
        return {
            "tp": legacy[1], "ts": legacy[2], "s_amp": legacy[3],
            "p_prob": legacy[4] if len(legacy) >= 6 else -1,
            "s_prob": legacy[5] if len(legacy) >= 6 else -1,
            "tp_std": 0.0, "ts_std": 0.0,
            "p_prob_std": 0.0, "s_prob_std": 0.0,
            "num_support": 1, "sources": [picker_name],
            "picker_cluster_sizes": {picker_name: 1},
        }
    sources = value("sources", [picker_name])
    if isinstance(sources, str):
        sources = [source for source in sources.split("|") if source]
    cluster_sizes = value("picker_cluster_sizes", "")
    if not cluster_sizes:
        cluster_sizes = {picker_name: int(value("num_support", 1))}
    return {
        "tp": value("tp"), "ts": value("ts"),
        "s_amp": value("s_amp", -1),
        "p_prob": value("p_prob", -1), "s_prob": value("s_prob", -1),
        "tp_std": value("tp_std", 0.0), "ts_std": value("ts_std", 0.0),
        "p_prob_std": value("p_prob_std", 0.0),
        "s_prob_std": value("s_prob_std", 0.0),
        "num_support": int(value("num_support", 1)),
        "sources": sources or [picker_name],
        "picker_cluster_sizes": cluster_sizes,
    }

def cleanup_sampling_rates(st, expected_sampling_rate=None, tolerance=0.01):
    """Drop duplicate-ID traces with bad sampling rates before ObsPy merge."""
    out = Stream()
    dropped = []
    grouped = {}
    for tr in st:
        grouped.setdefault(tr.id, []).append(tr)

    for trace_id, traces in grouped.items():
        rates = {}
        for tr in traces:
            rate_key = round(float(tr.stats.sampling_rate), 6)
            rates.setdefault(rate_key, []).append(tr)
        if len(rates) == 1:
            out += Stream(traces)
            continue

        if expected_sampling_rate:
            keep_rate = min(rates, key=lambda rate: abs(rate - expected_sampling_rate))
            if abs(keep_rate - expected_sampling_rate) > tolerance:
                keep_rate = max(rates, key=lambda rate: sum(len(tr.data) for tr in rates[rate]))
        else:
            keep_rate = max(rates, key=lambda rate: sum(len(tr.data) for tr in rates[rate]))

        for rate, rate_traces in rates.items():
            if rate == keep_rate:
                out += Stream(rate_traces)
            else:
                for tr in rate_traces:
                    dropped.append((trace_id, tr.stats.starttime, tr.stats.endtime, rate))

    for trace_id, t0, t1, rate in dropped:
        print("warning: drop bad sampling-rate trace {} {} {} rate {}".format(
            trace_id, t0, t1, rate
        ))
    return out, dropped


def read_realtime_mseed(mseed_path, expected_sampling_rate=None):
    """Read and merge an all-station miniSEED segment."""
    print("reading realtime mseed: {}".format(mseed_path))
    st = read(mseed_path)
    st, _ = cleanup_sampling_rates(st, expected_sampling_rate=expected_sampling_rate)
    st.merge(fill_value=0)
    return st


def read_realtime_mseed_timed(mseed_path, timing, expected_sampling_rate=None):
    """Read and merge an all-station miniSEED segment with timing."""
    print("reading realtime mseed: {}".format(mseed_path))
    t0 = time.perf_counter()
    try:
        st = read(mseed_path)
    except Exception as exc:
        raise BadRealtimeInputError(
            "cannot read {}: {}".format(mseed_path, exc)
        ) from exc
    timing["data_read_sec"] = time.perf_counter() - t0
    timing["num_raw_traces"] = len(st)

    t0 = time.perf_counter()
    try:
        st, dropped = cleanup_sampling_rates(
            st,
            expected_sampling_rate=expected_sampling_rate,
        )
    except Exception as exc:
        raise BadRealtimeInputError(
            "sampling-rate cleanup failed for {}: {}".format(
                mseed_path, exc
            )
        ) from exc
    timing["sampling_rate_cleanup_sec"] = time.perf_counter() - t0
    timing["num_bad_sampling_rate_traces"] = len(dropped)

    t0 = time.perf_counter()
    try:
        st.merge(fill_value=0)
    except Exception as exc:
        raise BadRealtimeInputError(
            "trace merge failed for {}: {}".format(mseed_path, exc)
        ) from exc
    timing["data_merge_sec"] = time.perf_counter() - t0
    timing["num_merged_traces"] = len(st)
    if len(st):
        # Filtering and tapering preserve these merged trace bounds. Medians
        # prevent a few late or short channels from redefining the segment.
        timing["data_start"] = format_time(median_time([
            tr.stats.starttime.datetime for tr in st
        ]))
        timing["data_end"] = format_time(median_time([
            (tr.stats.endtime + tr.stats.delta).datetime for tr in st
        ]))
    return st


def traces_for_station(st_all, sta_key, location_priority):
    net, sta, ch_prefix = parse_sta_key(sta_key)
    st_sel = st_all.select(network=net, station=sta)
    if ch_prefix:
        st_sel = Stream([tr for tr in st_sel if tr.stats.channel.startswith(ch_prefix)])
    if len(st_sel) == 0:
        return Stream()

    loc_groups = {}
    for tr in st_sel:
        loc = tr.stats.location or ""
        loc_groups.setdefault(loc, Stream()).append(tr)
    loc = choose_location(loc_groups, location_priority)
    return loc_groups[loc].copy()


def choose_location(loc_groups, location_priority):
    """Prefer configured borehole location codes when duplicate locs exist."""
    for loc in location_priority:
        if loc in loc_groups:
            return loc
    non_empty = sorted([loc for loc in loc_groups if loc])
    if non_empty:
        return non_empty[0]
    return sorted(loc_groups)[0]


def select_gain(gain, stream, station=None):
    """Select PAL-format gain for a stream midpoint."""
    start_time = max([tr.stats.starttime for tr in stream])
    end_time = min([tr.stats.endtime for tr in stream])
    st_time = start_time + (end_time - start_time) / 2
    return select_gain_for_time(
        gain, st_time, station=station, warn_fallback=True
    )


def apply_gain_and_units(stream, sta_key, sta_info):
    """Convert counts to velocity units expected by SAR/PAL magnitude."""
    gain = select_gain(sta_info[3], stream, station=sta_key)
    if isinstance(gain, float):
        gains = [gain, gain, gain]
    elif isinstance(gain, int):
        gains = [float(gain), float(gain), float(gain)]
    else:
        gains = [float(item) for item in gain]

    for ii, tr in enumerate(stream):
        if gains[ii] == 0:
            raise ValueError("zero gain for {} component {}".format(sta_key, ii))
        tr.data = tr.data.astype(np.float32, copy=False) / gains[ii]
        if tr.stats.channel.startswith("HN"):
            tr.detrend("demean")
            tr.integrate()
    return stream

def station_streams_from_mseed(st_all, sta_dict, location_priority=None):
    """Yield three-channel picker streams keyed by station selector."""
    if location_priority is None:
        location_priority = ["10", "20", "01", "00", ""]

    for sta_key in sorted(sta_dict):
        st = traces_for_station(st_all, sta_key, location_priority)
        if len(st) == 0:
            continue
        st = normalize_station_stream(st, sta_key, sta_dict[sta_key])
        if len(st) == 3:
            yield sta_key, st


def normalize_station_stream(st, sta_key, sta_info):
    """Convert any nonempty station/location group into three model channels."""
    by_comp = {}
    for tr in st:
        comp = tr.stats.channel[-1].upper()
        if comp in ("E", "N", "Z", "1", "2"):
            by_comp.setdefault(comp, tr)

    if "E" not in by_comp and "1" in by_comp:
        by_comp["E"] = by_comp["1"]
    if "N" not in by_comp and "2" in by_comp:
        by_comp["N"] = by_comp["2"]

    if all(comp in by_comp for comp in ("E", "N", "Z")):
        out = Stream([by_comp["E"].copy(), by_comp["N"].copy(), by_comp["Z"].copy()])
    else:
        # Preserve all available information without requiring conventional
        # component suffixes: [1, 2] -> [1, 2, 1], while longer groups use
        # their first three traces. Rename copies so downstream gain/unit and
        # picker code consistently see E/N/Z model channels.
        selected = (list(st) * 3)[:3]
        out = Stream()
        for trace, component in zip(selected, "ENZ"):
            normalized = trace.copy()
            prefix = normalized.stats.channel[:-1] if normalized.stats.channel else ""
            normalized.stats.channel = prefix + component
            out.append(normalized)
        if len(st) == 3:
            print(
                "warning: {} has 3 traces without a complete E/N/Z component "
                "set; using cyclic/truncated channel order {}".format(
                    sta_key, [tr.stats.channel for tr in st]
                )
            )

    net, sta, _ = parse_sta_key(sta_key)
    for tr in out:
        tr.stats.network = net
        tr.stats.station = sta
    return apply_gain_and_units(out, sta_key, sta_info)



def pick_segment(mseed_path, picker, sta_dict, out_pick_dir, location_priority=None,
                 vp=6.0, vs=3.45):
    """Run SAR picker for one miniSEED segment and save one .pick file."""
    if not os.path.exists(out_pick_dir):
        os.makedirs(out_pick_dir)

    pick_path = os.path.join(out_pick_dir, "{}.pick".format(segment_code(mseed_path)))
    st_all = read_realtime_mseed(mseed_path)
    picks = []

    with open(pick_path, "w") as fout:
        for sta_key, st in station_streams_from_mseed(st_all, sta_dict, location_priority):
            print("-" * 40)
            print("picking {}".format(sta_key))
            picks_i = picker.pick(st)
            if not picks_i:
                continue
            for pick in picks_i:
                record = unpack_picker_pick(
                    pick, getattr(picker, "name", picker.__class__.__name__.split("_")[0])
                )
                record["net_sta"] = sta_key
                tp, ts = record["tp"], record["ts"]
                sta_ot = calc_ot(tp, ts, vp=vp, vs=vs)
                picks.append((
                    sta_key, sta_ot, tp, ts, record["s_amp"],
                    record["p_prob"], record["s_prob"], record["tp_std"],
                    record["ts_std"], record["p_prob_std"],
                    record["s_prob_std"], record["num_support"],
                    "|".join(record["sources"]),
                    format_picker_cluster_sizes(record["picker_cluster_sizes"]),
                ))
                fout.write(format_pick_row(record))

    return np.array(picks, dtype=PICK_DTYPE), pick_path


def _timing_key(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def pick_segment_timed(mseed_path, pickers, sta_dict, pick_dirs,
                       location_priority=None, vp=6.0, vs=3.45,
                       num_workers=0, expected_sampling_rate=None,
                       preprocess_cfg=None, inference_executors=None):
    """Read one segment once and run every picker on each prepared station."""
    timing = {
        "data_read_sec": 0.0,
        "data_merge_sec": 0.0,
        "preprocess_sec": 0.0,
        "picking_sec": 0.0,
        "pick_write_sec": 0.0,
        "num_station_selectors": len(sta_dict),
        "num_station_streams": 0,
        "num_picker_workers": num_workers,
        "num_pickers": len(pickers),
        "waveform_spill_mb": 0.0,
    }
    pick_paths = {}
    outputs = {}
    picks_by_picker = {name: [] for name in pickers}
    retained_waveforms = {}
    for picker_name in pickers:
        out_dir = pick_dirs[picker_name]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pick_paths[picker_name] = os.path.join(
            out_dir, "{}.pick".format(segment_code(mseed_path))
        )
        outputs[picker_name] = open(pick_paths[picker_name], "w")

    st_all = read_realtime_mseed_timed(
        mseed_path,
        timing,
        expected_sampling_rate=expected_sampling_rate,
    )
    timing["rss_after_mseed_mb"] = process_rss_mb()
    if preprocess_cfg is None:
        raise ValueError("preprocess_cfg is required for shared realtime preprocessing")
    device_groups = _picker_device_groups(pickers)
    device_locks = {device: Lock() for device in device_groups}
    timing["num_picker_devices"] = len(device_groups)
    print("realtime picker device groups: {}".format({
        device: [name for name, _ in members]
        for device, members in device_groups.items()
    }))
    station_keys = sorted(sta_dict)
    segment = segment_code(mseed_path)
    waveform_spill_dir = os.path.join(
        preprocess_cfg.out_root,
        "_internal",
        "waveform_cache",
        segment,
    )
    # Initial associated-pick QC and optional postprocessing share this cache.
    shutil.rmtree(waveform_spill_dir, ignore_errors=True)
    os.makedirs(waveform_spill_dir, exist_ok=True)
    print(
        "filtered waveform QC cache: {}".format(waveform_spill_dir),
        flush=True,
    )
    memory_progress_interval = max(1, min(10, len(station_keys)))
    write_memory_progress(
        getattr(preprocess_cfg, "out_monitoring_dir", None),
        segment, 0, len(station_keys), timing["rss_after_mseed_mb"],
    )
    picking_wall_t0 = time.perf_counter()
    stations_completed = 0
    owns_station_executor = False
    owns_picker_executor = False
    station_executor = (
        inference_executors.station
        if inference_executors is not None else None
    )
    picker_executor = (
        inference_executors.picker
        if inference_executors is not None else None
    )
    if inference_executors is None and len(device_groups) > 1:
        picker_executor = ThreadPoolExecutor(
            max_workers=len(device_groups),
            thread_name_prefix="picker-device",
        )
        owns_picker_executor = True
    try:
        if num_workers and num_workers > 1:
            if station_executor is None:
                station_executor = ThreadPoolExecutor(
                    max_workers=num_workers,
                    thread_name_prefix="station-preprocess",
                )
                owns_station_executor = True
            result_iter = _bounded_station_results(
                station_executor,
                station_keys,
                num_workers,
                lambda sta_key: prepare_and_pick_station(
                    pickers, st_all, sta_key, sta_dict[sta_key], location_priority,
                    preprocess_cfg, device_groups, device_locks,
                    picker_executor=picker_executor,
                ),
            )
        else:
            result_iter = (
                prepare_and_pick_station(
                    pickers, st_all, sta_key, sta_dict[sta_key], location_priority,
                    preprocess_cfg, device_groups, device_locks,
                    picker_executor=picker_executor,
                )
                for sta_key in station_keys
            )

        for (
            sta_key, station_results, prep_sec, picker_seconds,
            has_stream, retained,
        ) in result_iter:
            stations_completed += 1
            timing["preprocess_sec"] += float(prep_sec)
            if has_stream:
                timing["num_station_streams"] += 1
            if retained is not None:
                if waveform_spill_dir is not None:
                    spill_name = re.sub(
                        r"[^A-Za-z0-9_.-]+", "_", sta_key
                    ) + ".pkl"
                    spill_path = os.path.join(waveform_spill_dir, spill_name)
                    retained.spill(spill_path)
                    timing["waveform_spill_mb"] += (
                        os.path.getsize(spill_path) / 1024.0 ** 2
                    )
                retained_waveforms[sta_key] = retained
            for picker_name in pickers:
                key = _timing_key(picker_name)
                timing["picking_{}_station_sum_sec".format(key)] = timing.get(
                    "picking_{}_station_sum_sec".format(key), 0.0
                ) + float(picker_seconds.get(picker_name, 0.0))
                picks_i = station_results.get(picker_name, [])
                if not picks_i:
                    continue
                t0 = time.perf_counter()
                for pick in picks_i:
                    record = unpack_picker_pick(pick, picker_name)
                    record["net_sta"] = sta_key
                    tp, ts = record["tp"], record["ts"]
                    sta_ot = calc_ot(tp, ts, vp=vp, vs=vs)
                    picks_by_picker[picker_name].append((
                        sta_key, sta_ot, tp, ts, record["s_amp"],
                        record["p_prob"], record["s_prob"], record["tp_std"],
                        record["ts_std"], record["p_prob_std"],
                        record["s_prob_std"], record["num_support"],
                        "|".join(record["sources"]),
                        format_picker_cluster_sizes(record["picker_cluster_sizes"]),
                    ))
                    outputs[picker_name].write(format_pick_row(record))
                timing["pick_write_sec"] += time.perf_counter() - t0
            if (
                stations_completed % memory_progress_interval == 0
                or stations_completed == len(station_keys)
            ):
                # Spilled filtered arrays have no remaining Python owner. Make
                # that release visible to the cgroup during this long stage.
                if waveform_spill_dir is not None:
                    trim_cpu_allocator()
                rss_mb = process_rss_mb()
                write_memory_progress(
                    getattr(preprocess_cfg, "out_monitoring_dir", None),
                    segment, stations_completed, len(station_keys), rss_mb,
                )
                print(
                    "station progress: {}/{} | RSS {:.1f} MB".format(
                        stations_completed, len(station_keys), rss_mb
                    ),
                    flush=True,
                )
    finally:
        if owns_station_executor:
            station_executor.shutdown(wait=True, cancel_futures=True)
        if owns_picker_executor:
            picker_executor.shutdown(wait=True, cancel_futures=True)
        for output in outputs.values():
            output.close()
    timing["picking_sec"] = time.perf_counter() - picking_wall_t0
    timing["rss_after_picking_mb"] = process_rss_mb()
    del st_all
    gc.collect()
    timing["rss_after_mseed_release_mb"] = process_rss_mb()

    arrays = {}
    for picker_name, picks in picks_by_picker.items():
        arrays[picker_name] = np.array(picks, dtype=PICK_DTYPE)
        timing["num_picks_{}".format(_timing_key(picker_name))] = len(picks)
    timing["num_picks"] = sum(len(picks) for picks in arrays.values())
    return arrays, pick_paths, timing, retained_waveforms


def build_association_branch_picks(
    picks_by_picker, pick_paths, segment, cfg, vp=6.0, vs=3.45,
):
    branch_picks = {}
    branch_pick_paths = {}
    for branch_name, members in cfg.association_branch_members.items():
        branch = cfg.result_branches[branch_name]
        if len(members) == 1 and branch_name != PREFERRED_ENSEMBLE_BRANCH:
            picker_name = members[0]
            branch_picks[branch_name] = picks_by_picker[picker_name]
            branch_pick_paths[branch_name] = pick_paths[picker_name]
            continue

        if branch_name == PREFERRED_ENSEMBLE_BRANCH:
            group_records = {}
            for group_name, group_members in (
                cfg.continuous_picker_groups.items()
            ):
                if not group_members:
                    continue
                records, _ = merge_picker_records(
                    {
                        name: picks_by_picker[name]
                        for name in group_members
                    },
                    cfg.tp_dev,
                    cfg.ts_dev,
                    min_support=(
                        cfg.continuous_picker_group_min_support[group_name]
                    ),
                )
                group_records[group_name] = records
            # Each group has already passed its own stability threshold.
            # The final preferred product is the union of accepted groups;
            # matching group picks are combined with equal group weight.
            records, input_counts = merge_picker_records(
                group_records, cfg.tp_dev, cfg.ts_dev, min_support=1
            )
            for record in records:
                record["sources"] = sorted(
                    record.get("picker_cluster_sizes", {})
                )
                record["num_support"] = len(record["sources"])
        else:
            records, input_counts = merge_picker_records(
                {name: picks_by_picker[name] for name in members},
                cfg.tp_dev,
                cfg.ts_dev,
                min_support=1,
            )
        os.makedirs(branch["pick_dir"], exist_ok=True)
        output_path = os.path.join(branch["pick_dir"], "{}.pick".format(segment))
        partial_path = output_path + ".partial"
        tuples = []
        with open(partial_path, "w") as fout:
            for record in records:
                tp, ts = UTCDateTime(record["tp"]), UTCDateTime(record["ts"])
                sta_ot = calc_ot(tp, ts, vp=vp, vs=vs)
                tuples.append((
                    record["net_sta"], sta_ot, tp, ts, record["s_amp"],
                    record["p_prob"], record["s_prob"], record["tp_std"],
                    record["ts_std"], record["p_prob_std"],
                    record["s_prob_std"], record["num_support"],
                    "|".join(record["sources"]),
                    format_picker_cluster_sizes(record["picker_cluster_sizes"]),
                ))
                fout.write(format_pick_row(record))
        os.replace(partial_path, output_path)
        branch_picks[branch_name] = np.array(tuples, dtype=ENSEMBLE_PICK_DTYPE)
        branch_pick_paths[branch_name] = output_path
        print("{} preferred ensemble: {} -> {} picks | {}".format(
            segment, input_counts, len(records), output_path
        ))
    return branch_picks, branch_pick_paths


def _record_event_repick_timing(timing, branch_key, summary, elapsed_sec):
    """Accumulate one segment-level repick/reassociation health record."""
    timing["event_repick_{}_sec".format(branch_key)] = elapsed_sec
    timing["event_repick_sec"] = timing.get("event_repick_sec", 0.0) + elapsed_sec
    if summary is None:
        return
    timing["num_event_repick_intervals"] = timing.get(
        "num_event_repick_intervals", 0
    ) + 1
    count_fields = {
        "num_event_repicker_pairs_generated": (
            "num_repicker_phase_pairs_generated"
        ),
        "num_event_repicker_pairs_glitch_rejected": (
            "num_repicker_phase_pairs_glitch_rejected"
        ),
        "num_event_repick_picks_reassoc_rejected": (
            "num_picks_reassociation_rejected"
        ),
        "num_event_repick_reassociated": "num_events_reassociated",
        "num_event_repick_reassociation_rejected": (
            "num_events_reassociation_rejected"
        ),
        "num_event_repick_late_s_skipped": "num_late_s_events_skipped",
        "num_event_repick_pairs_both_groups": "num_phase_pairs_both_groups",
        "num_event_repick_pairs_pos_neg_only": "num_phase_pairs_pos_neg_only",
        "num_event_repick_pairs_pos_only": "num_phase_pairs_pos_only",
        "num_event_repick_windows": "num_repick_windows",
        "num_event_repick_attempts": "num_station_event_attempts",
        "num_event_waveform_plots": "num_event_plots",
    }
    for timing_key, summary_key in count_fields.items():
        timing[timing_key] = timing.get(timing_key, 0) + int(
            summary.get(summary_key, 0)
        )
    for summary_key in (
        "job_build_sec", "window_prepare_sec", "device_inference_wall_sec",
        "device_transfer_sec", "result_merge_sec", "reassociation_sec",
        "waveform_qc_measurement_sec", "output_write_sec",
        "waveform_snapshot_sec", "plot_sec",
    ):
        timing_key = "event_repick_{}".format(summary_key)
        timing[timing_key] = timing.get(timing_key, 0.0) + float(
            summary.get(summary_key, 0.0)
        )
    for picker_name, seconds in summary.get("picker_seconds", {}).items():
        key = "event_repick_picker_{}_sec".format(_timing_key(picker_name))
        timing[key] = timing.get(key, 0.0) + float(seconds)

def process_segment(
    mseed_path, pickers, subnet_associators, pick_sta_dict, cfg,
    event_repick_coordinator=None, inference_executors=None,
):
    timing = {"segment": segment_code(mseed_path)}
    write_segment_memory_stage(
        cfg.out_monitoring_dir, timing["segment"], "segment_start"
    )
    segment_t0 = time.perf_counter()
    if event_repick_coordinator is not None:
        # Realtime repicking uses the current segment independently.
        # Release the preceding segment before reading/preparing another full
        # all-station waveform file.
        event_repick_coordinator.begin_segment(timing["segment"])
    pick_dirs = {
        name: cfg.picker_output_dirs[name] for name in pickers
    }
    picks_by_picker, pick_paths, pick_timing, retained_waveforms = pick_segment_timed(
        mseed_path,
        pickers,
        pick_sta_dict,
        pick_dirs,
        location_priority=cfg.location_priority,
        vp=cfg.vp,
        vs=cfg.vs,
        num_workers=cfg.num_workers,
        expected_sampling_rate=cfg.samp_rate,
        preprocess_cfg=cfg,
        inference_executors=inference_executors,
    )
    timing.update(pick_timing)
    write_segment_memory_stage(
        cfg.out_monitoring_dir, timing["segment"], "after_initial_picking"
    )

    segment = segment_code(mseed_path)
    if event_repick_coordinator is not None:
        event_repick_coordinator.register_segment(
            segment,
            timing["data_start"],
            timing["data_end"],
            retained_waveforms,
        )
    t0 = time.perf_counter()
    branch_picks, branch_pick_paths = build_association_branch_picks(
        picks_by_picker, pick_paths, segment, cfg, vp=cfg.vp, vs=cfg.vs
    )
    timing["picker_ensemble_sec"] = time.perf_counter() - t0
    selected_reference_branches = list(getattr(
        cfg, "event_waveform_plot_ref_branches", []
    ))
    branch_order = selected_reference_branches + [
        name for name in branch_picks if name not in selected_reference_branches
    ]
    for branch_name in branch_order:
        picks = branch_picks[branch_name]
        timing["num_picks_{}".format(_timing_key(branch_name))] = len(picks)
    phase_paths = {}
    timing["assoc_sec"] = 0.0
    timing["assoc_wall_sec"] = 0.0
    timing["merge_sec"] = 0.0
    timing["initial_waveform_qc_sec"] = 0.0
    timing["time_segment_merge_sec"] = 0.0
    timing["num_final_intervals_written"] = 0
    timing["num_final_events_written"] = 0
    initial_waveform_qc_cache = {}

    for branch_name, picks in branch_picks.items():
        branch = cfg.result_branches[branch_name]
        branch_key = _timing_key(branch_name)
        for out_dir in (
            branch.get("initial_phase_dir"),
            branch["phase_dir"], branch["subnet_phase_dir"],
            branch["merge_dir"], branch["final_cfg"].out_final_pha_dir,
            branch["final_cfg"].out_final_merge_dir, cfg.out_monitoring_dir,
        ):
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

        t0 = time.perf_counter()
        assoc_results = subnet_associators.associate(
            picks, segment, branch["result_name"], branch
        )
        assoc_wall = time.perf_counter() - t0
        timing["assoc_{}_wall_sec".format(branch_key)] = assoc_wall
        timing["assoc_wall_sec"] += assoc_wall
        subnet_phase_paths = []
        assoc_sum = 0.0
        for subnet_name, result in sorted(assoc_results.items()):
            assoc_sum += result["assoc_sec"]
            timing["assoc_{}_{}_sec".format(branch_key, subnet_name)] = result["assoc_sec"]
            timing["picks_{}_{}".format(branch_key, subnet_name)] = result["num_picks"]
            timing["events_{}_{}".format(branch_key, subnet_name)] = len(
                read_phase_file(result["pha_path"])
            )
            subnet_phase_paths.append(result["pha_path"])
        timing["assoc_{}_station_sum_sec".format(branch_key)] = assoc_sum
        timing["assoc_sec"] += assoc_sum

        merged_phase_path = os.path.join(
            branch["phase_dir"], "phase_{}.dat".format(segment)
        )
        merge_log_path = os.path.join(
            branch["merge_dir"], "merge_{}.csv".format(segment)
        )
        t0 = time.perf_counter()
        if len(subnet_phase_paths) == 1:
            # No duplicate reconciliation is needed for one full-network result.
            shutil.copyfile(subnet_phase_paths[0], merged_phase_path)
            num_events = len(read_phase_file(merged_phase_path))
            with open(merge_log_path, "w", newline="") as fout:
                writer = csv.writer(fout)
                writer.writerow(["mode", "input_phase", "num_events"])
                writer.writerow(["single_network", subnet_phase_paths[0], num_events])
            merge_summary = {"num_merged_events": num_events}
        else:
            merge_summary = merge_phase_files(
                subnet_phase_paths,
                merged_phase_path,
                merge_log_path,
                origin_time_tol_sec=cfg.merge_origin_time_tol_sec,
                epicenter_tol_km=cfg.merge_epicenter_tol_km,
                depth_tol_km=cfg.merge_depth_tol_km,
                min_shared_phase_stations=cfg.merge_min_shared_phase_stations,
                phase_pick_time_tol_sec=cfg.merge_phase_pick_time_tol_sec,
                time_format_digits=cfg.merge_time_format_digits,
            )
        merge_sec = time.perf_counter() - t0
        timing["merge_{}_sec".format(branch_key)] = merge_sec
        timing["merge_sec"] += merge_sec
        timing["merged_events_{}".format(branch_key)] = merge_summary["num_merged_events"]
        write_segment_memory_stage(
            cfg.out_monitoring_dir,
            segment,
            "after_subnet_merge_{}".format(branch_key),
        )

        t0 = time.perf_counter()
        initial_qc = qc_initial_phase_file(
            merged_phase_path,
            retained_waveforms,
            pick_sta_dict,
            cfg,
            cache=initial_waveform_qc_cache,
        )
        initial_qc_sec = time.perf_counter() - t0
        timing["initial_waveform_qc_{}_sec".format(branch_key)] = initial_qc_sec
        timing["initial_waveform_qc_sec"] += initial_qc_sec
        timing["initial_glitch_picks_rejected_{}".format(branch_key)] = (
            initial_qc["num_rejected_picks"]
        )
        timing["initial_events_qc_rejected_{}".format(branch_key)] = (
            initial_qc["num_rejected_events"]
        )
        print(
            "{} initial waveform QC: {} -> {} events | {} glitch picks "
            "rejected".format(
                branch_name,
                initial_qc["num_input_events"],
                initial_qc["num_output_events"],
                initial_qc["num_rejected_picks"],
            ),
            flush=True,
        )

        initial_phase_dir = branch.get("initial_phase_dir")
        if initial_phase_dir:
            initial_phase_path = os.path.join(
                initial_phase_dir, "phase_{}.dat".format(segment)
            )
            partial_initial_phase_path = initial_phase_path + ".partial"
            shutil.copyfile(
                merged_phase_path, partial_initial_phase_path
            )
            os.replace(partial_initial_phase_path, initial_phase_path)
            timing[
                "initial_events_{}".format(branch_key)
            ] = len(read_phase_file(initial_phase_path))

        if (
            event_repick_coordinator is not None
            and branch_name == PREFERRED_ENSEMBLE_BRANCH
        ):
            t0 = time.perf_counter()
            repick_summary = event_repick_coordinator.process_segment_result(
                branch_name, segment, merged_phase_path, force=True
            )
            _record_event_repick_timing(
                timing, branch_key, repick_summary,
                time.perf_counter() - t0,
            )
            write_segment_memory_stage(
                cfg.out_monitoring_dir,
                segment,
                "after_event_repick_{}".format(branch_key),
            )

        if (
            event_repick_coordinator is not None
            and branch_name in getattr(
                cfg, "event_waveform_plot_ref_branches", []
            )
        ):
            t0 = time.perf_counter()
            num_reference_snapshots = (
                event_repick_coordinator.capture_reference_result(
                    branch_name, segment, merged_phase_path
                )
            )
            timing[
                "num_event_waveform_snapshots_{}".format(branch_key)
            ] = num_reference_snapshots
            timing[
                "event_waveform_snapshot_{}_sec".format(branch_key)
            ] = time.perf_counter() - t0

        merged_events = read_phase_file(merged_phase_path)
        timing["merged_events_{}".format(branch_key)] = len(merged_events)
        if merged_events:
            merged_ot_min = min(event["time"] for event in merged_events)
            merged_ot_max = max(event["time"] for event in merged_events)
            data_start_dt = (
                parse_time(timing["data_start"])
                if timing.get("data_start") else None
            )
            data_end_dt = (
                parse_time(timing["data_end"])
                if timing.get("data_end") else None
            )
            num_events_in_waveform = sum(
                (data_start_dt is None or event["time"] >= data_start_dt)
                and (data_end_dt is None or event["time"] < data_end_dt)
                for event in merged_events
            )
            print(
                "{} postprocessed origin range: {} -- {} | {} of {} "
                "inside waveform [{} -- {})".format(
                    branch_name,
                    format_time(merged_ot_min), format_time(merged_ot_max),
                    num_events_in_waveform, len(merged_events),
                    format_time(data_start_dt) if data_start_dt else "-inf",
                    format_time(data_end_dt) if data_end_dt else "+inf",
                ),
                flush=True,
            )
            if num_events_in_waveform == 0:
                print(
                    "warning: {} reassociated origins do not overlap the "
                    "measured waveform window".format(branch_name),
                    flush=True,
                )
        # Count unique station P/S pairs so a duplicated phase row cannot make
        # the association ratio exceed the branch input-pick population.
        num_associated_picks = len({
            (pick["sta"], pick["p"], pick["s"])
            for event in merged_events for pick in event["picks"]
        })
        timing["num_associated_picks_{}".format(
            branch_key
        )] = num_associated_picks
        timing["assoc_ratio_{}".format(branch_key)] = (
            float(num_associated_picks) / len(picks) if len(picks) else 0.0
        )
        phase_paths[branch_name] = merged_phase_path

        t0 = time.perf_counter()
        try:
            final_results = update_final_merged_outputs(
                segment,
                merged_phase_path,
                timing.get("data_start"),
                timing.get("data_end"),
                branch["final_cfg"],
            )
            timing["time_segment_merge_failed_{}".format(branch_key)] = 0
        except Exception as exc:
            final_results = []
            timing["time_segment_merge_failed_{}".format(branch_key)] = 1
            print("warning: segment finalization failed for {} {} | {}: {}".format(
                branch_name, segment, exc.__class__.__name__, exc
            ))
        # update_final_merged_outputs rewrites the internal phase file after
        # same-segment deduplication and corrected-origin filtering.
        merged_events = read_phase_file(merged_phase_path)
        timing["merged_events_{}".format(branch_key)] = len(merged_events)
        num_associated_picks = len({
            (pick["sta"], pick["p"], pick["s"])
            for event in merged_events for pick in event["picks"]
        })
        timing["num_associated_picks_{}".format(
            branch_key
        )] = num_associated_picks
        timing["assoc_ratio_{}".format(branch_key)] = (
            float(num_associated_picks) / len(picks) if len(picks) else 0.0
        )
        final_sec = time.perf_counter() - t0
        timing["time_segment_merge_{}_sec".format(branch_key)] = final_sec
        timing["time_segment_merge_sec"] += final_sec
        timing["num_final_intervals_written"] += len(final_results)
        reported_final_events = sum(
            int(result.get("num_merged_events", 0))
            for result in final_results
        )
        # The phase files are the published product and therefore the
        # authoritative event-count source.  A merge summary can describe
        # candidates before the final interval write/filtering is complete.
        branch_final_events = sum(
            len(read_phase_file(result["phase_path"]))
            for result in final_results
        )
        if reported_final_events != branch_final_events:
            print(
                "warning: {} final merge reported {} events, but the "
                "published phase files contain {}".format(
                    branch_name, reported_final_events,
                    branch_final_events,
                ),
                flush=True,
            )
        timing["final_events_{}".format(branch_key)] = branch_final_events
        timing["num_final_events_written"] += branch_final_events
        if event_repick_coordinator is not None and (
            branch_name == PREFERRED_ENSEMBLE_BRANCH
            or branch_name in getattr(
                cfg, "event_waveform_plot_ref_branches", []
            )
        ):
            final_plot_summary = event_repick_coordinator.plot_final_results(
                final_results, branch_name=branch_name
            )
            plotted_final_events = int(
                final_plot_summary.get("num_final_events", 0)
            )
            timing[
                "event_waveform_final_events_{}".format(branch_key)
            ] = plotted_final_events
            timing[
                "event_waveform_final_plots_{}".format(branch_key)
            ] = int(final_plot_summary["num_plots"])
            timing[
                "event_waveform_final_missing_snapshot_{}".format(branch_key)
            ] = int(final_plot_summary.get("num_missing_snapshots", 0))
            timing[
                "event_waveform_final_unusable_waveform_{}".format(branch_key)
            ] = int(final_plot_summary.get("num_unrendered_waveforms", 0))
            if plotted_final_events != branch_final_events:
                print(
                    "warning: {} final phase count ({}) differs from the "
                    "waveform plotting input count ({})".format(
                        branch_name, branch_final_events,
                        plotted_final_events,
                    ),
                    flush=True,
                )
            timing["num_event_waveform_final_plots"] = timing.get(
                "num_event_waveform_final_plots", 0
            ) + int(final_plot_summary["num_plots"])
            timing["event_waveform_final_plot_sec"] = timing.get(
                "event_waveform_final_plot_sec", 0.0
            ) + float(final_plot_summary["plot_sec"])
            timing["event_repick_plot_sec"] = timing.get(
                "event_repick_plot_sec", 0.0
            ) + float(final_plot_summary["plot_sec"])
            timing["num_event_waveform_plots"] = timing.get(
                "num_event_waveform_plots", 0
            ) + int(final_plot_summary["num_plots"])
            timing["num_event_waveform_snapshots_released"] = timing.get(
                "num_event_waveform_snapshots_released", 0
            ) + int(final_plot_summary["released"])
            timing["num_event_waveform_snapshots_pending"] = int(
                timing.get("num_event_waveform_snapshots_pending", 0)
                + final_plot_summary.get("pending", 0)
            )

    if event_repick_coordinator is not None:
        event_repick_coordinator.release_segment_cache(
            "after preferred and selected reference snapshots for {}".format(
                segment
            )
        )
    else:
        for holder in retained_waveforms.values():
            holder.release()
        retained_waveforms.clear()
        trim_cpu_allocator()

    t0 = time.perf_counter()
    write_picker_selection_state(segment, cfg)
    timing["selection_state_write_sec"] = time.perf_counter() - t0
    timing["rss_segment_end_mb"] = process_rss_mb()
    write_segment_memory_stage(
        cfg.out_monitoring_dir, segment, "segment_end"
    )
    timing["total_sec"] = time.perf_counter() - segment_t0
    return pick_paths, phase_paths, picks_by_picker, timing

def parse_time(value):
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def format_time(value, time_format_digits=2):
    if hasattr(value, "datetime"):
        value = value.datetime
    value = value + timedelta(microseconds=5000)
    value = value.replace(microsecond=(value.microsecond // 10000) * 10000)
    text = value.isoformat(timespec="milliseconds")
    return text[:-1] + "Z"


def median_time(times):
    base = min(times)
    offsets = [(item - base).total_seconds() for item in times]
    return base + timedelta(seconds=median(offsets))



def median_valid(values, default=-1, min_value=None):
    valid = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        if min_value is not None and value < min_value:
            continue
        valid.append(value)
    if not valid:
        return default
    return median(valid)
def horizontal_distance_km(lat1, lon1, lat2, lon2):
    lat0 = 0.5 * (lat1 + lat2)
    dx = (lon2 - lon1) * 111.32 * np.cos(np.deg2rad(lat0))
    dy = (lat2 - lat1) * 111.32
    return float(np.hypot(dx, dy))


def is_event_header(codes):
    if len(codes) != 5 or "T" not in codes[0]:
        return False
    try:
        float(codes[1])
        float(codes[2])
        float(codes[3])
        float(codes[4])
    except ValueError:
        return False
    return True


def read_phase_file(fpha):
    events = []
    current = None

    with open(fpha) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            codes = [code.strip() for code in line.split(",")]
            if is_event_header(codes):
                if current is not None:
                    events.append(current)
                current = {
                    "source": fpha,
                    "time": parse_time(codes[0]),
                    "lat": float(codes[1]),
                    "lon": float(codes[2]),
                    "depth": float(codes[3]),
                    "mag": float(codes[4]),
                    "picks": [],
                }
            else:
                if current is None:
                    raise ValueError("Pick row before event header in {}: {}".format(fpha, line))
                if len(codes) < 4:
                    raise ValueError("Bad pick row in {}: {}".format(fpha, line))
                current["picks"].append(
                    {
                        "sta": codes[0],
                        "p": parse_time(codes[1]),
                        "s": parse_time(codes[2]),
                        "score": float(codes[3]),
                        "p_prob": float(codes[4]) if len(codes) > 4 else -1,
                        "s_prob": float(codes[5]) if len(codes) > 5 else -1,
                        "tp_std": float(codes[6]) if len(codes) > 6 else 0.0,
                        "ts_std": float(codes[7]) if len(codes) > 7 else 0.0,
                        "p_prob_std": float(codes[8]) if len(codes) > 8 else 0.0,
                        "s_prob_std": float(codes[9]) if len(codes) > 9 else 0.0,
                        "num_support": int(codes[10]) if len(codes) > 10 else 1,
                        "pickers": codes[11] if len(codes) > 11 else "",
                        "sources": codes[11] if len(codes) > 11 else "",
                        "picker_cluster_sizes": (
                            codes[12] if len(codes) > 12 else ""
                        ),
                        "picker_uncertainties": (
                            codes[13] if len(codes) > 13 else ""
                        ),
                        "pick_provenance": (
                            codes[14] if len(codes) > 14 else "initial"
                        ),
                        "repick_status": (
                            codes[15] if len(codes) > 15 else "unknown"
                        ),
                        "repick_support": (
                            int(codes[16])
                            if len(codes) > 16 and codes[16] else -1
                        ),
                        "repick_sources": (
                            codes[17] if len(codes) > 17 else ""
                        ),
                        "repick_required_support": (
                            int(codes[18])
                            if len(codes) > 18 and codes[18] else -1
                        ),
                        "p_snr_e": (
                            float(codes[19]) if len(codes) > 19 else -1.0
                        ),
                        "p_snr_n": (
                            float(codes[20]) if len(codes) > 20 else -1.0
                        ),
                        "p_snr_z": (
                            float(codes[21]) if len(codes) > 21 else -1.0
                        ),
                    }
                )

    if current is not None:
        events.append(current)
    return events

def _probability_lookup_from_picks(picks):
    lookup = {}
    if len(picks) == 0 or "p_prob" not in picks.dtype.names or "s_prob" not in picks.dtype.names:
        return lookup
    for pick in picks:
        lookup.setdefault(pick["net_sta"], []).append(
            {
                "tp": pick["tp"],
                "ts": pick["ts"],
                "p_prob": float(pick["p_prob"]),
                "s_prob": float(pick["s_prob"]),
            }
        )
    return lookup


def _match_pick_probability(codes, lookup, tolerance_sec=0.001):
    sta = codes[0]
    if sta not in lookup:
        return None
    try:
        tp = UTCDateTime(codes[1])
        ts = UTCDateTime(codes[2])
    except Exception:
        return None

    best = None
    best_dt = None
    for item in lookup[sta]:
        dt = abs(tp - item["tp"]) + abs(ts - item["ts"])
        if best_dt is None or dt < best_dt:
            best = item
            best_dt = dt
    if best is None or best_dt > tolerance_sec:
        return None
    return best["p_prob"], best["s_prob"]


def enrich_phase_probabilities(fpha, picks, tolerance_sec=0.001):
    """Fill p_prob/s_prob columns in a PAL phase file from picker output."""
    lookup = _probability_lookup_from_picks(picks)
    if not lookup or not os.path.exists(fpha):
        return 0

    num_filled = 0
    out_lines = []
    with open(fpha) as fp:
        for line in fp:
            raw = line.rstrip("\n")
            if not raw.strip():
                out_lines.append(line)
                continue
            codes = [code.strip() for code in raw.split(",")]
            if is_event_header(codes) or len(codes) < 4:
                out_lines.append(line)
                continue
            probs = _match_pick_probability(codes, lookup, tolerance_sec=tolerance_sec)
            if probs is None:
                out_lines.append(line)
                continue
            p_prob, s_prob = probs
            needs_fill = len(codes) < 6
            if not needs_fill:
                try:
                    needs_fill = float(codes[4]) < 0 or float(codes[5]) < 0
                except ValueError:
                    needs_fill = True
            if not needs_fill:
                out_lines.append(line)
                continue
            if len(codes) < 5:
                codes.append("{:.4f}".format(p_prob))
            else:
                codes[4] = "{:.4f}".format(p_prob)
            if len(codes) < 6:
                codes.append("{:.4f}".format(s_prob))
            else:
                codes[5] = "{:.4f}".format(s_prob)
            out_lines.append(",".join(codes) + "\n")
            num_filled += 1

    with open(fpha, "w") as fp:
        fp.writelines(out_lines)
    return num_filled

def events_match(event_a, event_b, origin_time_tol_sec, epicenter_tol_km, depth_tol_km):
    dt = abs((event_b["time"] - event_a["time"]).total_seconds())
    if dt > origin_time_tol_sec:
        return False
    dist = horizontal_distance_km(
        event_a["lat"], event_a["lon"], event_b["lat"], event_b["lon"]
    )
    if dist > epicenter_tol_km:
        return False
    ddepth = abs(event_b["depth"] - event_a["depth"])
    return ddepth <= depth_tol_km


def group_events(events, origin_time_tol_sec, epicenter_tol_km, depth_tol_km,
                 min_shared_phase_stations=0,
                 phase_pick_time_tol_sec=1.0):
    """Group duplicate detections as connected components of pairwise matches."""
    num_events = len(events)
    if num_events == 0:
        return []

    parent = list(range(num_events))

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    # Link by phase picks first. This is independent of origin time and event
    # location, which may be biased when subnets use different station subsets.
    if min_shared_phase_stations > 0:
        entries_by_station = {}
        for event_idx, event in enumerate(events):
            for pick in event["picks"]:
                entries_by_station.setdefault(pick["sta"], {}).setdefault(
                    event_idx, []
                ).append(pick)

        matched_station_counts = {}
        for event_picks in entries_by_station.values():
            event_indices = sorted(event_picks)
            for left_pos, left_idx in enumerate(event_indices):
                for right_idx in event_indices[left_pos + 1:]:
                    matched = any(
                        abs((right_pick["p"] - left_pick["p"]).total_seconds())
                        < phase_pick_time_tol_sec
                        and abs((right_pick["s"] - left_pick["s"]).total_seconds())
                        < phase_pick_time_tol_sec
                        for left_pick in event_picks[left_idx]
                        for right_pick in event_picks[right_idx]
                    )
                    if matched:
                        pair = (left_idx, right_idx)
                        matched_station_counts[pair] = (
                            matched_station_counts.get(pair, 0) + 1
                        )

        for pair, num_stations in matched_station_counts.items():
            if num_stations >= min_shared_phase_stations:
                union(pair[0], pair[1])

    indexed_events = sorted(enumerate(events), key=lambda item: item[1]["time"])
    for left_pos, (left_idx, left_event) in enumerate(indexed_events):
        right_pos = left_pos + 1
        while right_pos < num_events:
            right_idx, right_event = indexed_events[right_pos]
            dt = (right_event["time"] - left_event["time"]).total_seconds()
            if dt > origin_time_tol_sec:
                break
            if events_match(
                left_event, right_event,
                origin_time_tol_sec, epicenter_tol_km, depth_tol_km,
            ):
                union(left_idx, right_idx)
            right_pos += 1

    groups_by_root = {}
    for idx, event in enumerate(events):
        groups_by_root.setdefault(find(idx), []).append(event)

    groups = []
    for group_events_i in groups_by_root.values():
        group_events_i = sorted(group_events_i, key=lambda item: item["time"])
        groups.append({
            "events": group_events_i,
            "last_time": group_events_i[-1]["time"],
        })
    return sorted(groups, key=lambda group: median_time([event["time"] for event in group["events"]]))


def merge_group(group, phase_pick_time_tol_sec=1.0):
    events = group["events"]
    merged = {
        "time": median_time([event["time"] for event in events]),
        "lat": median([event["lat"] for event in events]),
        "lon": median([event["lon"] for event in events]),
        "depth": median([event["depth"] for event in events]),
        "mag": median_valid([event["mag"] for event in events], default=-1, min_value=0),
        "picks": [],
        "num_events": len(events),
        "sources": sorted({event["source"] for event in events}),
    }

    picks_by_sta = {}
    for event in events:
        for pick in event["picks"]:
            picks_by_sta.setdefault(pick["sta"], []).append(pick)

    for sta in sorted(picks_by_sta):
      for phase_group in _cluster_station_picks(
          picks_by_sta[sta], phase_pick_time_tol_sec
      ):
        provenance, picks = select_preferred_provenance_picks(phase_group)
        merged["picks"].append(
            {
                "sta": sta,
                "p": median_time([pick["p"] for pick in picks]),
                "s": median_time([pick["s"] for pick in picks]),
                "score": median([pick["score"] for pick in picks]),
                "p_prob": median_valid([pick["p_prob"] for pick in picks], default=-1, min_value=0),
                "s_prob": median_valid([pick["s_prob"] for pick in picks], default=-1, min_value=0),
                "tp_std": median([pick["tp_std"] for pick in picks]),
                "ts_std": median([pick["ts_std"] for pick in picks]),
                "p_prob_std": median([pick["p_prob_std"] for pick in picks]),
                "s_prob_std": median([pick["s_prob_std"] for pick in picks]),
                "num_support": max(pick["num_support"] for pick in picks),
                "pickers": "|".join(sorted({
                    picker
                    for pick in picks
                    for picker in pick["pickers"].split("|")
                    if picker
                })),
                "picker_cluster_sizes": format_picker_cluster_sizes(
                    merge_picker_cluster_sizes(
                        pick["picker_cluster_sizes"] for pick in picks
                    )
                ),
                "picker_uncertainties": "|".join(sorted({
                    value
                    for pick in picks
                    for value in pick.get("picker_uncertainties", "").split("|")
                    if value
                })),
                "pick_provenance": provenance,
                "repick_status": "|".join(sorted({
                    value
                    for pick in picks
                    for value in pick.get("repick_status", "unknown").split("|")
                    if value
                })),
                "repick_support": max(
                    pick.get("repick_support", -1) for pick in picks
                ),
                "repick_sources": "|".join(sorted({
                    source
                    for pick in picks
                    for source in pick.get("repick_sources", "").split("|")
                    if source
                })),
                "repick_required_support": max(
                    pick.get("repick_required_support", -1) for pick in picks
                ),
                "p_snr_e": median_valid([
                    pick.get("p_snr_e", -1.0) for pick in picks
                ], default=-1, min_value=0),
                "p_snr_n": median_valid([
                    pick.get("p_snr_n", -1.0) for pick in picks
                ], default=-1, min_value=0),
                "p_snr_z": median_valid([
                    pick.get("p_snr_z", -1.0) for pick in picks
                ], default=-1, min_value=0),
                "num_picks": len(picks),
            }
        )

    return merged


def merge_phase_files(fpha_list, fpha_out, fmerge_log, origin_time_tol_sec=2.5,
                      epicenter_tol_km=5.0, depth_tol_km=10.0,
                      time_format_digits=6, event_time_start=None,
                      event_time_end=None, exclude_phase_files=None,
                      min_shared_phase_stations=0,
                      phase_pick_time_tol_sec=1.0,
                      min_both_group_ratio=None):
    events = []
    file_event_counts = {}
    for fpha in sorted(fpha_list):
        file_events = read_phase_file(fpha)
        file_event_counts[fpha] = len(file_events)
        events.extend(file_events)
        print("{}: {} events".format(fpha, len(file_events)))

    num_input_events = len(events)
    num_input_events_in_interval = sum(
        (event_time_start is None or event["time"] >= event_time_start)
        and (event_time_end is None or event["time"] < event_time_end)
        for event in events
    )
    for fpha in exclude_phase_files or []:
        if not fpha or not os.path.exists(fpha):
            continue
        for event in read_phase_file(fpha):
            event["exclude_from_output"] = True
            events.append(event)

    groups = group_events(
        events,
        origin_time_tol_sec,
        epicenter_tol_km,
        depth_tol_km,
        min_shared_phase_stations=min_shared_phase_stations,
        phase_pick_time_tol_sec=phase_pick_time_tol_sec,
    )
    groups = [
        group for group in groups
        if not any(
            event.get("exclude_from_output", False)
            for event in group["events"]
        )
    ]
    num_grouped_events = len(groups)
    merged_events = []
    for group in groups:
        merged = merge_group(group, phase_pick_time_tol_sec)
        # Assign a duplicate group to exactly one disjoint final interval by
        # its canonical merged origin. Using "any member in interval" can put
        # a boundary-straddling group in two final files and previously led to
        # a fragile exclusion against the prior generated output.
        if (
            event_time_start is not None
            and merged["time"] < event_time_start
        ):
            continue
        if event_time_end is not None and merged["time"] >= event_time_end:
            continue
        merged_events.append(merged)
    merged_events = sorted(merged_events, key=lambda item: item["time"])
    num_interval_events_before_both_group_qc = len(merged_events)
    num_both_group_ratio_rejected = 0
    if min_both_group_ratio is not None:
        accepted = [
            event for event in merged_events
            if event_both_group_pick_ratio(event) >= min_both_group_ratio
        ]
        num_both_group_ratio_rejected = len(merged_events) - len(accepted)
        merged_events = accepted

    with open(fpha_out, "w") as fp:
        for event in merged_events:
            fp.write(
                "{},{:.5f},{:.5f},{:.1f},{:.2f}\n".format(
                    format_time(event["time"], time_format_digits),
                    event["lat"],
                    event["lon"],
                    event["depth"],
                    event["mag"],
                )
            )
            for pick in event["picks"]:
                fp.write(
                    "{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},"
                    "{:.4f},{:.4f},{},{},{},{},{},{},{},{},{},"
                    "{:.4f},{:.4f},{:.4f}\n".format(
                        pick["sta"],
                        format_time(pick["p"], time_format_digits),
                        format_time(pick["s"], time_format_digits),
                        pick["score"], pick["p_prob"], pick["s_prob"],
                        pick["tp_std"], pick["ts_std"],
                        pick["p_prob_std"], pick["s_prob_std"],
                        pick["num_support"], pick["pickers"],
                        pick["picker_cluster_sizes"],
                        pick.get("picker_uncertainties", ""),
                        pick.get("pick_provenance", "initial"),
                        pick.get("repick_status", "unknown"),
                        pick.get("repick_support", -1),
                        pick.get("repick_sources", ""),
                        pick.get("repick_required_support", -1),
                        pick.get("p_snr_e", -1.0),
                        pick.get("p_snr_n", -1.0),
                        pick.get("p_snr_z", -1.0),
                    )
                )

    with open(fmerge_log, "w") as fp:
        fp.write("merged_event_id,num_input_events,num_sources,num_picks,time,lat,lon,depth,mag,sources\n")
        for event_id, event in enumerate(merged_events):
            fp.write(
                "{},{},{},{},{},{:.5f},{:.5f},{:.1f},{:.2f},{}\n".format(
                    event_id,
                    event["num_events"],
                    len(event["sources"]),
                    len(event["picks"]),
                    format_time(event["time"], time_format_digits),
                    event["lat"],
                    event["lon"],
                    event["depth"],
                    event["mag"],
                    "|".join(event["sources"]),
                )
            )

    num_multi_event_groups = sum(1 for event in merged_events if event["num_events"] > 1)
    max_input_events_per_group = max([event["num_events"] for event in merged_events], default=0)
    summary = {
        "num_input_phase_files": len(fpha_list),
        "num_input_events": num_input_events,
        "num_input_events_in_final_interval": num_input_events_in_interval,
        "num_merged_events": len(merged_events),
        "num_events_both_group_ratio_rejected": num_both_group_ratio_rejected,
        "num_duplicate_events_removed": num_input_events - num_grouped_events,
        "num_events_outside_final_interval": (
            num_grouped_events - num_interval_events_before_both_group_qc
        ),
        "num_multi_input_event_groups": num_multi_event_groups,
        "max_input_events_per_group": max_input_events_per_group,
    }
    print("input phase files: {}".format(summary["num_input_phase_files"]))
    print("input events: {}".format(summary["num_input_events"]))
    if event_time_start is not None or event_time_end is not None:
        print(
            "input events in final interval [{} -- {}): {}".format(
                format_time(event_time_start) if event_time_start else "-inf",
                format_time(event_time_end) if event_time_end else "+inf",
                summary["num_input_events_in_final_interval"],
            )
        )
        for fpha in sorted(fpha_list):
            source_events = read_phase_file(fpha)
            if source_events:
                print(
                    "  {} origin range: {} -- {} | {} in interval".format(
                        fpha,
                        format_time(min(event["time"] for event in source_events)),
                        format_time(max(event["time"] for event in source_events)),
                        sum(
                            (event_time_start is None or event["time"] >= event_time_start)
                            and (event_time_end is None or event["time"] < event_time_end)
                            for event in source_events
                        ),
                    )
                )
    print("merged events: {}".format(summary["num_merged_events"]))
    if min_both_group_ratio is not None:
        print(
            "events rejected below both-repicker-group ratio {:.3f}: {}".format(
                min_both_group_ratio,
                summary["num_events_both_group_ratio_rejected"],
            )
        )
    print("duplicate input events removed: {}".format(summary["num_duplicate_events_removed"]))
    print("multi-input-event groups: {}".format(summary["num_multi_input_event_groups"]))
    print("max input events per merged group: {}".format(summary["max_input_events_per_group"]))
    print("output phase file: {}".format(fpha_out))
    print("merge log: {}".format(fmerge_log))
    return summary


TIME_SEGMENT_MERGE_VERSION = "12_corrected_segment_cursor"
SEGMENT_WINDOW_FIELDS = [
    "segment", "start", "end", "phase_path", "bounds_source",
]
FINALIZED_WINDOW_FIELDS = [
    "merge_version", "segment", "valid_start", "valid_end",
    "report_start", "report_end", "phase_path", "catalog_path",
    "merge_log_path", "num_events",
]


def _write_csv_atomic(path, fieldnames, rows):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_path, path)


def _load_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))


def _segment_timestamp_from_name(segment):
    match = re.search(r"(\d{8}T\d{6})Z?", segment)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def _time_token(value):
    return value.strftime("%Y%m%dT%H%M%SZ")


def _load_segment_windows(path):
    records = []
    for row in _load_csv_rows(path):
        try:
            start = parse_time(row["start"])
            end = parse_time(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        phase_path = row.get("phase_path", "")
        if end <= start or not phase_path:
            continue
        records.append({
            "segment": row["segment"],
            "start": start,
            "end": end,
            "phase_path": phase_path,
            "bounds_source": row.get("bounds_source", "legacy"),
        })
    return records


def _save_segment_windows(path, records):
    rows = []
    for record in sorted(records, key=lambda item: item["start"]):
        rows.append({
            "segment": record["segment"],
            "start": format_time(record["start"]),
            "end": format_time(record["end"]),
            "phase_path": record["phase_path"],
            "bounds_source": record.get("bounds_source", "legacy"),
        })
    _write_csv_atomic(path, SEGMENT_WINDOW_FIELDS, rows)


def _waveform_bounds(mseed_path, expected_sampling_rate=None):
    """Read, clean, and merge miniSEED to recover canonical segment bounds."""
    st = read(mseed_path)
    st, _ = cleanup_sampling_rates(
        st, expected_sampling_rate=expected_sampling_rate
    )
    st.merge(fill_value=0)
    if len(st) == 0:
        return None
    start = median_time([tr.stats.starttime.datetime for tr in st])
    end = median_time([
        (tr.stats.endtime + tr.stats.delta).datetime for tr in st
    ])
    if end <= start:
        return None
    return start, end


def _phase_files_by_segment(cfg):
    phase_files = {}
    for phase_path in glob.glob(os.path.join(cfg.out_pha_dir, "phase_*.dat")):
        stem = os.path.splitext(os.path.basename(phase_path))[0]
        if not stem.startswith("phase_"):
            continue
        phase_files[stem[6:]] = os.path.abspath(phase_path)
    return phase_files


def _mseed_files_by_segment(cfg):
    return {
        segment_code(path): path
        for path in glob.glob(os.path.join(cfg.in_dir, cfg.in_glob))
    }


def _fallback_window_duration(segment, starts_by_segment, known_durations):
    start = starts_by_segment[segment]
    starts = sorted(set(starts_by_segment.values()))
    position = starts.index(start)
    strides = []
    if position > 0:
        strides.append(start - starts[position - 1])
    if position + 1 < len(starts):
        strides.append(starts[position + 1] - start)
    strides = [stride for stride in strides if stride.total_seconds() > 0]
    if strides:
        # Realtime windows overlap by half their length, so duration is twice
        # the local arrival stride. The shorter neighbor avoids treating a gap
        # in the available archive as a longer data window.
        return min(strides) * 2
    if known_durations:
        seconds = median([duration.total_seconds() for duration in known_durations])
        return timedelta(seconds=seconds)
    return None


def _discover_existing_segment_windows(cfg, refresh_waveforms=True):
    """Build window metadata for every original phase file currently in OUT."""
    phase_files = _phase_files_by_segment(cfg)
    if not phase_files:
        return []

    existing = {
        record["segment"]: record
        for record in _load_segment_windows(cfg.segment_window_path)
        if record["segment"] in phase_files
    }
    mseed_files = _mseed_files_by_segment(cfg) if refresh_waveforms else {}
    records = {}
    measured = 0
    preserved = 0

    for segment, phase_path in phase_files.items():
        if (
            not refresh_waveforms
            and segment in existing
            and existing[segment].get("bounds_source") == "inferred_end_timestamp"
        ):
            record = dict(existing[segment])
            record["phase_path"] = phase_path
            records[segment] = record
            preserved += 1
            continue

        mseed_path = mseed_files.get(segment)
        if refresh_waveforms and mseed_path:
            try:
                bounds = _waveform_bounds(mseed_path)
            except Exception as exc:
                bounds = None
                print("warning: cannot read headers for {} | {}: {}".format(
                    mseed_path, exc.__class__.__name__, exc
                ))
            if bounds is not None:
                records[segment] = {
                    "segment": segment,
                    "start": bounds[0],
                    "end": bounds[1],
                    "phase_path": phase_path,
                    "bounds_source": "waveform",
                }
                measured += 1
                continue

        if (
            segment in existing
            and existing[segment].get("bounds_source") == "inferred_end_timestamp"
        ):
            record = dict(existing[segment])
            record["phase_path"] = phase_path
            records[segment] = record
            preserved += 1

    # SCSN realtime filenames encode the window end (for example, events in
    # phase_scsn_...T103001Z occur in the segment ending near 10:30:01).
    ends_by_segment = {
        segment: _segment_timestamp_from_name(segment)
        for segment in phase_files
    }
    ends_by_segment = {
        segment: end for segment, end in ends_by_segment.items()
        if end is not None
    }
    known_durations = [
        record["end"] - record["start"] for record in records.values()
    ]
    inferred = 0
    skipped = 0
    for segment, phase_path in phase_files.items():
        if segment in records:
            continue
        if segment not in ends_by_segment:
            skipped += 1
            print("warning: cannot infer phase window end from {}".format(phase_path))
            continue
        duration = _fallback_window_duration(
            segment, ends_by_segment, known_durations
        )
        if duration is None or duration.total_seconds() <= 0:
            skipped += 1
            print("warning: cannot infer phase window duration for {}".format(phase_path))
            continue
        end = ends_by_segment[segment]
        records[segment] = {
            "segment": segment,
            "start": end - duration,
            "end": end,
            "phase_path": phase_path,
            "bounds_source": "inferred_end_timestamp",
        }
        inferred += 1

    records = sorted(records.values(), key=lambda item: item["start"])
    _save_segment_windows(cfg.segment_window_path, records)
    print(
        "discovered {} existing phase windows: {} waveform, {} saved, "
        "{} inferred, {} skipped".format(
            len(records), measured, preserved, inferred, skipped
        )
    )
    return records

def _register_segment_window(segment, phase_path, data_start, data_end, cfg):
    if not data_start or not data_end:
        print("warning: no waveform bounds for {}; final time merge skipped".format(segment))
        return _discover_existing_segment_windows(cfg, refresh_waveforms=False)

    start = parse_time(data_start)
    end = parse_time(data_end)
    if end <= start:
        print("warning: invalid waveform bounds for {}; final time merge skipped".format(segment))
        return _discover_existing_segment_windows(cfg, refresh_waveforms=False)

    records = {
        record["segment"]: record
        for record in _load_segment_windows(cfg.segment_window_path)
    }
    records[segment] = {
        "segment": segment,
        "start": start,
        "end": end,
        "phase_path": os.path.abspath(phase_path),
        "bounds_source": "waveform",
    }
    _save_segment_windows(cfg.segment_window_path, list(records.values()))

    # Discover any phase files that appeared before this process started or
    # outside the normal picker callback, while preserving known exact bounds.
    return _discover_existing_segment_windows(cfg, refresh_waveforms=False)

def _publish_final_interval(previous, current, interval_start, interval_end, cfg):
    interval_code = "{}_{}".format(
        _time_token(interval_start), _time_token(interval_end)
    )
    phase_path = os.path.join(
        cfg.out_final_pha_dir, "phase_final_{}.dat".format(interval_code)
    )
    write_catalog = getattr(cfg, "write_catalog_outputs", True)
    catalog_path = (
        os.path.join(
            cfg.out_final_ctlg_dir, "catalog_final_{}.dat".format(interval_code)
        )
        if write_catalog else ""
    )
    merge_log_path = os.path.join(
        cfg.out_final_merge_dir, "merge_final_{}.csv".format(interval_code)
    )
    out_dirs = [cfg.out_final_pha_dir, cfg.out_final_merge_dir]
    if write_catalog:
        out_dirs.append(cfg.out_final_ctlg_dir)
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    phase_tmp = phase_path + ".tmp"
    catalog_tmp = catalog_path + ".tmp" if write_catalog else None
    merge_log_tmp = merge_log_path + ".tmp"
    source_phase_paths = sorted(set(
        previous.get("phase_paths", [previous.get("phase_path")])
        + current.get("phase_paths", [current.get("phase_path")])
    ))
    source_phase_paths = [path for path in source_phase_paths if path]
    summary = merge_phase_files(
        source_phase_paths,
        phase_tmp,
        merge_log_tmp,
        origin_time_tol_sec=cfg.merge_origin_time_tol_sec,
        epicenter_tol_km=cfg.merge_epicenter_tol_km,
        depth_tol_km=cfg.merge_depth_tol_km,
        min_shared_phase_stations=cfg.merge_min_shared_phase_stations,
        phase_pick_time_tol_sec=cfg.merge_phase_pick_time_tol_sec,
        time_format_digits=cfg.merge_time_format_digits,
        event_time_start=interval_start,
        event_time_end=interval_end,
        # The both-repicker-group QC is applied to each reassociated candidate
        # in EventRepicker. Publication must not reinterpret provenance after
        # duplicate merging or reject the accepted events a second time.
        min_both_group_ratio=None,
    )
    if write_catalog:
        write_catalog_from_phase(phase_tmp, catalog_tmp)

    if (
        summary["num_input_events_in_final_interval"] > 0
        and summary["num_merged_events"] == 0
        and summary["num_events_both_group_ratio_rejected"] == 0
    ):
        for temporary in (phase_tmp, catalog_tmp, merge_log_tmp):
            if temporary and os.path.exists(temporary):
                os.remove(temporary)
        raise RuntimeError(
            "final merge suppressed every in-interval source event without "
            "a both-group QC rejection"
        )

    os.replace(merge_log_tmp, merge_log_path)
    if write_catalog:
        os.replace(catalog_tmp, catalog_path)
    os.replace(phase_tmp, phase_path)
    summary.update({
        "phase_path": phase_path,
        "catalog_path": catalog_path,
        "merge_log_path": merge_log_path,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "waveform_segments": sorted(set(
            previous.get("segments", []) + current.get("segments", [])
        )),
    })
    print(
        "final interval {} to {}: {} events | {}".format(
            format_time(interval_start), format_time(interval_end),
            summary["num_merged_events"], phase_path,
        )
    )
    return summary

def _phase_endpoint_groups(records, cfg):
    """Cluster source files representing the same logical window endpoint."""
    endpoint_records = []
    for record in records:
        endpoint = _segment_timestamp_from_name(record["segment"])
        if endpoint is None:
            endpoint = record["end"]
        endpoint_records.append((endpoint, record))
    endpoint_records.sort(key=lambda item: item[0])

    tolerance_sec = max(5.0, 2.0 * cfg.merge_origin_time_tol_sec)
    raw_groups = []
    for endpoint, record in endpoint_records:
        if (
            not raw_groups
            or (endpoint - raw_groups[-1][-1][0]).total_seconds() > tolerance_sec
        ):
            raw_groups.append([])
        raw_groups[-1].append((endpoint, record))

    groups = []
    for raw_group in raw_groups:
        endpoint = median_time([item[0] for item in raw_group])
        records_i = [item[1] for item in raw_group]
        groups.append({
            "id": "endpoint_{}".format(_time_token(endpoint)),
            "end": endpoint,
            # Same-endpoint records are alternate phase products for one
            # waveform window. Median bounds tolerate sub-second channel
            # differences while preserving the measured window geometry.
            "window_start": median_time([item["start"] for item in records_i]),
            "window_end": median_time([item["end"] for item in records_i]),
            "phase_paths": sorted({item["phase_path"] for item in records_i}),
            "segments": sorted({item["segment"] for item in records_i}),
        })
    return groups


def _typical_endpoint_stride_sec(groups):
    deltas = [
        (groups[idx]["end"] - groups[idx - 1]["end"]).total_seconds()
        for idx in range(1, len(groups))
    ]
    deltas = [delta for delta in deltas if delta > 0]
    if not deltas:
        return None
    return float(median(deltas))

def _remove_obsolete_final_outputs(cfg):
    """Remove only generated final products from an obsolete merge version."""
    patterns = (
        (cfg.out_final_pha_dir, "phase_final_*.dat"),
        (cfg.out_final_ctlg_dir, "catalog_final_*.dat"),
        (cfg.out_final_merge_dir, "merge_final_*.csv"),
    )
    removed = 0
    for out_dir, pattern in patterns:
        root = os.path.abspath(out_dir)
        for path in glob.glob(os.path.join(out_dir, pattern)):
            absolute_path = os.path.abspath(path)
            try:
                inside_root = os.path.commonpath([root, absolute_path]) == root
            except ValueError:
                inside_root = False
            if inside_root and os.path.isfile(absolute_path):
                os.remove(absolute_path)
                removed += 1
    if removed:
        print("removed {} obsolete generated final files".format(removed))


def _load_current_finalized_rows(cfg):
    all_rows = _load_csv_rows(cfg.finalized_window_path)
    current_rows = [
        row for row in all_rows
        if row.get("merge_version") == TIME_SEGMENT_MERGE_VERSION
        and os.path.exists(row.get("phase_path", ""))
        and (
            not getattr(cfg, "write_catalog_outputs", True)
            or os.path.exists(row.get("catalog_path", ""))
        )
    ]
    generated_phase_files = glob.glob(os.path.join(
        cfg.out_final_pha_dir, "phase_final_*.dat"
    ))
    if (
        len(current_rows) != len(all_rows)
        or (not current_rows and generated_phase_files)
    ):
        _remove_obsolete_final_outputs(cfg)
        current_rows = []
    _write_csv_atomic(
        cfg.finalized_window_path,
        FINALIZED_WINDOW_FIELDS,
        current_rows,
    )
    return current_rows


def _legacy_finalize_segment_windows(records, cfg):
    """Publish one disjoint final interval per adjacent endpoint pair."""
    finalized_rows = _load_current_finalized_rows(cfg)
    endpoint_groups = _phase_endpoint_groups(records, cfg)
    if len(endpoint_groups) < 2:
        print("final time merge waiting for the next overlapping segment")
        return []

    stride_sec = _typical_endpoint_stride_sec(endpoint_groups)
    if stride_sec is None or stride_sec <= 0:
        print("warning: cannot determine realtime phase-window stride")
        return []
    print(
        "final merge geometry: filename timestamp=end | {:.2f}s stride | "
        "{:.2f}s nominal source window | {} endpoint groups".format(
            stride_sec, 2.0 * stride_sec, len(endpoint_groups)
        )
    )

    finalized_by_pair = {
        (row.get("previous_segment"), row.get("current_segment")): row
        for row in finalized_rows
    }
    finalized_pairs = set(finalized_by_pair)
    results = []

    for idx in range(1, len(endpoint_groups)):
        previous = endpoint_groups[idx - 1]
        current = endpoint_groups[idx]
        endpoint_delta = (current["end"] - previous["end"]).total_seconds()
        if endpoint_delta > 1.5 * stride_sec:
            print(
                "warning: skip missing-window gap {} to {} ({:.2f}s)".format(
                    previous["id"], current["id"], endpoint_delta
                )
            )
            continue

        # Finalize the actual common waveform coverage. Filename timestamps
        # identify logical neighbors, but the saved waveform bounds define
        # which origin times are valid when acquisition timestamps drift.
        interval_start = max(
            previous["window_start"], current["window_start"]
        )
        interval_end = min(previous["window_end"], current["window_end"])
        if interval_end <= interval_start:
            print(
                "warning: skip non-overlapping waveform windows {} and {}: "
                "{} -- {}".format(
                    previous["id"], current["id"],
                    format_time(interval_start), format_time(interval_end),
                )
            )
            continue
        pair = (previous["id"], current["id"])

        if pair in finalized_pairs:
            continue
        if not all(
            os.path.exists(path)
            for path in previous["phase_paths"] + current["phase_paths"]
        ):
            continue

        summary = _publish_final_interval(
            previous,
            current,
            interval_start,
            interval_end,
            cfg,
        )
        finalized_rows.append({
            "merge_version": TIME_SEGMENT_MERGE_VERSION,
            "start": format_time(interval_start),
            "end": format_time(interval_end),
            "phase_path": summary["phase_path"],
            "catalog_path": summary["catalog_path"],
            "merge_log_path": summary["merge_log_path"],
            "previous_segment": previous["id"],
            "current_segment": current["id"],
            "num_events": summary["num_merged_events"],
        })
        _write_csv_atomic(
            cfg.finalized_window_path,
            FINALIZED_WINDOW_FIELDS,
            finalized_rows,
        )
        finalized_pairs.add(pair)
        finalized_by_pair[pair] = finalized_rows[-1]
        results.append(summary)

    return results

def _legacy_update_final_merged_outputs(segment, phase_path, data_start, data_end, cfg):
    """Register a new phase window and publish newly finalized intervals."""
    records = _register_segment_window(
        segment, phase_path, data_start, data_end, cfg
    )
    return _legacy_finalize_segment_windows(records, cfg)


def _legacy_bootstrap_final_merged_outputs(cfg, eligible_phase_paths=None):
    """Finalize all pending original phase files before realtime polling."""
    t0 = time.perf_counter()
    out_dirs = [
        cfg.out_final_pha_dir,
        cfg.out_final_merge_dir,
        cfg.out_monitoring_dir,
    ]
    if getattr(cfg, "write_catalog_outputs", True):
        out_dirs.append(cfg.out_final_ctlg_dir)
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    eligible = (
        {os.path.abspath(path) for path in eligible_phase_paths}
        if eligible_phase_paths is not None else None
    )
    phase_files = _phase_files_by_segment(cfg)
    if eligible is not None:
        phase_files = {
            segment: path for segment, path in phase_files.items()
            if os.path.abspath(path) in eligible
        }
    num_phase_files = len(phase_files)
    print(
        "startup final merge: found {} original phase files in {}".format(
            num_phase_files, cfg.out_pha_dir
        ),
        flush=True,
    )
    # Historical startup must be fast. Infer the half-overlap window geometry
    # from phase filename timestamps; do not parse large all-station miniSEEDs.
    records = _discover_existing_segment_windows(cfg, refresh_waveforms=False)
    if eligible is not None:
        records = [
            record for record in records
            if os.path.abspath(record["phase_path"]) in eligible
        ]
        print(
            "startup final merge: {} phase windows verified as repicked"
            .format(len(records)),
            flush=True,
        )
    results = _legacy_finalize_segment_windows(records, cfg)
    elapsed = time.perf_counter() - t0
    num_events = sum(result["num_merged_events"] for result in results)
    print(
        "startup final time merge: {:.2f}s | {} phase windows | "
        "{} final intervals written".format(
            elapsed, len(records), len(results)
        ),
        flush=True,
    )

    if not os.path.exists(cfg.out_monitoring_dir):
        os.makedirs(cfg.out_monitoring_dir)
    timing_path = os.path.join(
        cfg.out_monitoring_dir, "timing_final_merge_startup_multi_picker.csv"
    )
    write_header = not os.path.exists(timing_path)
    with open(timing_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow([
                "time", "result_name", "num_phase_windows", "num_final_intervals_written",
                "num_final_events_written", "startup_final_merge_sec",
            ])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            getattr(cfg, "result_name", "default"),
            len(records),
            len(results),
            num_events,
            elapsed,
        ])
    return results


# Corrected-origin realtime publication. These definitions supersede the
# legacy adjacent-window merger above while keeping old monitoring CSVs
# readable during a version transition.
def _corrected_origin_bounds(record, cfg):
    taper_sec = float(getattr(cfg, "taper_max_length_sec", 0.0))
    association_buffer_sec = float(
        getattr(cfg, "association_buffer_sec", 0.0)
    )
    start = record["start"] + timedelta(seconds=taper_sec)
    end = record["end"] - timedelta(
        seconds=taper_sec + association_buffer_sec
    )
    if end <= start:
        raise ValueError(
            "empty corrected origin interval for {}: {} -- {}".format(
                record["segment"], format_time(start), format_time(end)
            )
        )
    return start, end


def _merge_corrected_interval(source_paths, phase_path, merge_log_path,
                              interval_start, interval_end, cfg):
    phase_tmp = phase_path + ".tmp"
    merge_log_tmp = merge_log_path + ".tmp"
    summary = merge_phase_files(
        source_paths,
        phase_tmp,
        merge_log_tmp,
        origin_time_tol_sec=cfg.merge_origin_time_tol_sec,
        epicenter_tol_km=cfg.merge_epicenter_tol_km,
        depth_tol_km=cfg.merge_depth_tol_km,
        min_shared_phase_stations=cfg.merge_min_shared_phase_stations,
        phase_pick_time_tol_sec=cfg.merge_phase_pick_time_tol_sec,
        time_format_digits=cfg.merge_time_format_digits,
        event_time_start=interval_start,
        event_time_end=interval_end,
        # EventRepicker already applies this QC to reassociated candidates.
        # The output stage only deduplicates and filters corrected OT bounds.
        min_both_group_ratio=None,
    )
    os.replace(merge_log_tmp, merge_log_path)
    os.replace(phase_tmp, phase_path)
    return summary


def _normalize_corrected_segment(record, cfg):
    """Merge same-segment duplicates, then apply [T0_corr, T1_corr)."""
    valid_start, valid_end = _corrected_origin_bounds(record, cfg)
    merge_log_path = os.path.join(
        cfg.out_final_merge_dir,
        "merge_segment_{}.csv".format(record["segment"]),
    )
    for out_dir in (cfg.out_final_merge_dir, os.path.dirname(record["phase_path"])):
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
    summary = _merge_corrected_interval(
        [record["phase_path"]], record["phase_path"], merge_log_path,
        valid_start, valid_end, cfg,
    )
    normalized = dict(record)
    normalized.update({
        "valid_start": valid_start,
        "valid_end": valid_end,
        "segment_merge_log_path": merge_log_path,
    })
    print(
        "segment {} valid origin interval [{} -- {}): {} events".format(
            record["segment"], format_time(valid_start),
            format_time(valid_end), summary["num_merged_events"],
        ),
        flush=True,
    )
    return normalized


def _publish_corrected_tail(record, report_start, report_end, cfg):
    interval_code = "{}_{}".format(
        _time_token(report_start), _time_token(report_end)
    )
    phase_path = os.path.join(
        cfg.out_final_pha_dir, "phase_final_{}.dat".format(interval_code)
    )
    merge_log_path = os.path.join(
        cfg.out_final_merge_dir, "merge_final_{}.csv".format(interval_code)
    )
    write_catalog = getattr(cfg, "write_catalog_outputs", True)
    catalog_path = (
        os.path.join(
            cfg.out_final_ctlg_dir,
            "catalog_final_{}.dat".format(interval_code),
        ) if write_catalog else ""
    )
    out_dirs = [cfg.out_final_pha_dir, cfg.out_final_merge_dir]
    if write_catalog:
        out_dirs.append(cfg.out_final_ctlg_dir)
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    summary = _merge_corrected_interval(
        [record["phase_path"]], phase_path, merge_log_path,
        report_start, report_end, cfg,
    )
    if write_catalog:
        catalog_tmp = catalog_path + ".tmp"
        write_catalog_from_phase(phase_path, catalog_tmp)
        os.replace(catalog_tmp, catalog_path)
    summary.update({
        "phase_path": phase_path,
        "catalog_path": catalog_path,
        "merge_log_path": merge_log_path,
        "interval_start": report_start,
        "interval_end": report_end,
        "waveform_segments": [record["segment"]],
    })
    print(
        "final reported interval [{} -- {}): {} events | {}".format(
            format_time(report_start), format_time(report_end),
            summary["num_merged_events"], phase_path,
        ),
        flush=True,
    )
    return summary


def _finalize_segment_windows(records, cfg):
    """Publish only the portion of each valid segment beyond the OT cursor."""
    finalized_rows = _load_current_finalized_rows(cfg)
    finalized_segments = {row.get("segment") for row in finalized_rows}
    cursor = (
        max(parse_time(row["report_end"]) for row in finalized_rows)
        if finalized_rows else None
    )
    normalized_records = []
    for record in records:
        if os.path.exists(record["phase_path"]):
            normalized_records.append(_normalize_corrected_segment(record, cfg))

    results = []
    for record in sorted(normalized_records, key=lambda item: item["valid_end"]):
        if record["segment"] in finalized_segments:
            continue
        report_start = record["valid_start"]
        report_end = record["valid_end"]
        if cursor is not None:
            if report_end <= cursor:
                print(
                    "segment {} adds no coverage beyond final cursor {}".format(
                        record["segment"], format_time(cursor)
                    ),
                    flush=True,
                )
                continue
            if report_start <= cursor:
                report_start = cursor
            else:
                print(
                    "warning: realtime origin-time coverage gap [{} -- {})"
                    .format(format_time(cursor), format_time(report_start)),
                    flush=True,
                )

        summary = _publish_corrected_tail(
            record, report_start, report_end, cfg
        )
        finalized_rows.append({
            "merge_version": TIME_SEGMENT_MERGE_VERSION,
            "segment": record["segment"],
            "valid_start": format_time(record["valid_start"]),
            "valid_end": format_time(record["valid_end"]),
            "report_start": format_time(report_start),
            "report_end": format_time(report_end),
            "phase_path": summary["phase_path"],
            "catalog_path": summary["catalog_path"],
            "merge_log_path": summary["merge_log_path"],
            "num_events": summary["num_merged_events"],
        })
        _write_csv_atomic(
            cfg.finalized_window_path,
            FINALIZED_WINDOW_FIELDS,
            finalized_rows,
        )
        finalized_segments.add(record["segment"])
        cursor = report_end
        results.append(summary)
    return results


def _register_exact_segment_window(segment, phase_path, data_start, data_end, cfg):
    if not data_start or not data_end:
        raise ValueError("missing measured waveform bounds for {}".format(segment))
    start = parse_time(data_start)
    end = parse_time(data_end)
    if end <= start:
        raise ValueError("invalid measured waveform bounds for {}".format(segment))
    records = {
        record["segment"]: record
        for record in _load_segment_windows(cfg.segment_window_path)
        if record.get("bounds_source") != "inferred_end_timestamp"
    }
    records[segment] = {
        "segment": segment,
        "start": start,
        "end": end,
        "phase_path": os.path.abspath(phase_path),
        "bounds_source": "median_merged_traces",
    }
    _save_segment_windows(cfg.segment_window_path, list(records.values()))
    return records[segment]


def _discover_exact_segment_windows(cfg):
    phase_files = _phase_files_by_segment(cfg)
    saved = {
        record["segment"]: record
        for record in _load_segment_windows(cfg.segment_window_path)
        if record.get("bounds_source") != "inferred_end_timestamp"
    }
    mseed_files = _mseed_files_by_segment(cfg)
    records = []
    for segment, phase_path in sorted(phase_files.items()):
        if segment in saved:
            record = dict(saved[segment])
            record["phase_path"] = phase_path
            records.append(record)
            continue
        mseed_path = mseed_files.get(segment)
        if mseed_path is None:
            print(
                "warning: skip startup phase {} because exact waveform bounds "
                "are unavailable".format(phase_path),
                flush=True,
            )
            continue
        try:
            bounds = _waveform_bounds(
                mseed_path,
                expected_sampling_rate=getattr(cfg, "samp_rate", None),
            )
        except Exception as exc:
            print(
                "warning: cannot measure waveform bounds for {} | {}: {}"
                .format(mseed_path, exc.__class__.__name__, exc),
                flush=True,
            )
            continue
        if bounds is None:
            continue
        records.append({
            "segment": segment,
            "start": bounds[0],
            "end": bounds[1],
            "phase_path": phase_path,
            "bounds_source": "median_merged_traces",
        })
    _save_segment_windows(cfg.segment_window_path, records)
    return records


def update_final_merged_outputs(segment, phase_path, data_start, data_end, cfg):
    """Normalize one segment and advance the external reporting cursor."""
    record = _register_exact_segment_window(
        segment, phase_path, data_start, data_end, cfg
    )
    return _finalize_segment_windows([record], cfg)


def bootstrap_final_merged_outputs(cfg, eligible_phase_paths=None):
    """Backfill corrected segment products and monotonically reported tails."""
    t0 = time.perf_counter()
    eligible = (
        {os.path.abspath(path) for path in eligible_phase_paths}
        if eligible_phase_paths is not None else None
    )
    records = _discover_exact_segment_windows(cfg)
    if eligible is not None:
        records = [
            record for record in records
            if os.path.abspath(record["phase_path"]) in eligible
        ]
    print(
        "startup corrected-segment finalization: {} eligible phase windows"
        .format(len(records)),
        flush=True,
    )
    results = _finalize_segment_windows(records, cfg)
    elapsed = time.perf_counter() - t0
    num_events = sum(result["num_merged_events"] for result in results)
    if not os.path.exists(cfg.out_monitoring_dir):
        os.makedirs(cfg.out_monitoring_dir)
    timing_path = os.path.join(
        cfg.out_monitoring_dir, "timing_final_merge_startup_multi_picker.csv"
    )
    write_header = not os.path.exists(timing_path)
    with open(timing_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow([
                "time", "result_name", "num_phase_windows",
                "num_final_intervals_written", "num_final_events_written",
                "startup_final_merge_sec",
            ])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            getattr(cfg, "result_name", "default"), len(records),
            len(results), num_events, elapsed,
        ])
    print(
        "startup corrected-segment finalization: {:.2f}s | {} intervals | "
        "{} events".format(elapsed, len(results), num_events),
        flush=True,
    )
    return results


def pending_segment_repick_results(cfg, completed_phase_paths=None):
    """Return original merged segment phases not yet repicked/reassociated."""
    completed = {
        os.path.abspath(path) for path in (completed_phase_paths or set())
    }
    return [
        {"segment": segment, "phase_path": phase_path}
        for segment, phase_path in sorted(_phase_files_by_segment(cfg).items())
        if os.path.abspath(phase_path) not in completed
    ]


def write_catalog_from_phase(fpha, fctlg):
    with open(fpha) as fin, open(fctlg, "w") as fout:
        for line in fin:
            codes = [code.strip() for code in line.split(",")]
            if is_event_header(codes):
                fout.write(line)


def write_timing_report(out_dir, segment, timing):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ftime = os.path.join(out_dir, "timing_{}.csv".format(segment))
    keys = sorted(timing)
    with open(ftime, "w") as fp:
        fp.write(",".join(keys) + "\n")
        fp.write(",".join(str(timing[key]) for key in keys) + "\n")

    fall = os.path.join(out_dir, "timing_all_multi_picker.csv")
    write_header = not os.path.exists(fall)
    with open(fall, "a") as fp:
        if write_header:
            fp.write(",".join(keys) + "\n")
        fp.write(",".join(str(timing.get(key, "")) for key in keys) + "\n")
    return ftime


def load_done_records(done_record_path):
    if not os.path.exists(done_record_path):
        return set()
    with open(done_record_path) as fp:
        return {line.strip() for line in fp if line.strip()}


def picker_selection_state_path(segment, cfg):
    return os.path.join(
        cfg.picker_selection_state_dir, "{}.json".format(segment)
    )


def write_picker_selection_state(segment, cfg):
    os.makedirs(cfg.picker_selection_state_dir, exist_ok=True)
    state_path = picker_selection_state_path(segment, cfg)
    partial_path = state_path + ".partial"
    with open(partial_path, "w", encoding="utf-8") as fp:
        json.dump(
            {"picker_selection_signature": cfg.picker_selection_signature},
            fp,
            sort_keys=True,
        )
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(partial_path, state_path)


def picker_selection_state_matches(segment, cfg):
    state_path = picker_selection_state_path(segment, cfg)
    try:
        with open(state_path, encoding="utf-8") as fp:
            state = json.load(fp)
    except (OSError, ValueError, TypeError):
        return False
    return state.get("picker_selection_signature") == (
        cfg.picker_selection_signature
    )


def pipeline_outputs_complete(mseed_path, cfg):
    """Return true only when every picker and association branch is complete."""
    segment = segment_code(mseed_path)
    if not picker_selection_state_matches(segment, cfg):
        return False
    for picker_name in cfg.enabled_pickers:
        pick_path = os.path.join(
            cfg.picker_output_dirs[picker_name], "{}.pick".format(segment)
        )
        if not os.path.isfile(pick_path):
            return False

    for branch_name in cfg.association_branches:
        branch = cfg.result_branches[branch_name]
        pick_path = os.path.join(branch["pick_dir"], "{}.pick".format(segment))
        phase_path = os.path.join(
            branch["phase_dir"], "phase_{}.dat".format(segment)
        )
        if not os.path.isfile(pick_path) or not os.path.isfile(phase_path):
            return False
    return True

def append_done_record(done_record_path, mseed_path):
    out_dir = os.path.dirname(done_record_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    record = os.path.abspath(mseed_path)
    with open(done_record_path, "a") as fp:
        fp.write(record + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    return record


def load_bad_records(bad_record_path):
    if not os.path.exists(bad_record_path):
        return set()
    records = set()
    with open(bad_record_path, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            path = row.get("path")
            if path:
                records.add(path)
    return records


def append_bad_record(bad_record_path, mseed_path, exc):
    out_dir = os.path.dirname(bad_record_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    write_header = not os.path.exists(bad_record_path)
    record = os.path.abspath(mseed_path)
    with open(bad_record_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(["time", "path", "error_type", "error"])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            record,
            exc.__class__.__name__,
            str(exc),
        ])
        fp.flush()
        os.fsync(fp.fileno())
    return record

def plot_timing_report(plot_script, timing_csv, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plot_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(timing_csv))[0] + ".png",
    )
    result = subprocess.run(
        [sys.executable, plot_script, timing_csv, plot_path],
        check=False,
    )
    if result.returncode != 0:
        print("warning: timing plot failed with code {}".format(result.returncode))
        return None
    print("timing plot: {}".format(plot_path))
    return plot_path


def write_initialization_timing(out_dir, startup_timing, model_load_sec=0.0,
                                time_table_wall_sec=0.0,
                                model_load_times=None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_load_times = model_load_times or {"model": model_load_sec}
    model_names = sorted(model_load_times)
    model_columns = [
        "model_load_{}_sec".format(_timing_key(name)) for name in model_names
    ]
    path = os.path.join(out_dir, "timing_initialization.csv")
    with open(path, "w") as fp:
        fp.write(",".join(
            ["subnet", "num_station_selectors", "time_table_sec"]
            + model_columns + ["time_table_wall_sec"]
        ) + "\n")
        for subnet, result in sorted(startup_timing.items()):
            fp.write(",".join(str(value) for value in (
                [subnet, result["num_stations"], result["time_table_sec"]]
                + [model_load_times[name] for name in model_names]
                + [time_table_wall_sec]
            )) + "\n")

def print_timing_report(timing):
    print("-" * 60)
    print("timing summary for {}".format(timing["segment"]))
    for key in sorted(timing):
        if key.endswith("_sec") or key.endswith("_mb") or key.startswith("num_"):
            print("  {}: {}".format(key, timing[key]))


def write_realtime_heartbeat(cfg, state, start_time, processed, num_input_files,
                             num_pending, num_complete, num_bad,
                             worker_status=""):
    """Append a durable liveness record for operations and postmortem checks."""
    if not os.path.exists(cfg.out_monitoring_dir):
        os.makedirs(cfg.out_monitoring_dir)
    path = os.path.join(cfg.out_monitoring_dir, "realtime_heartbeat.csv")
    fields = [
        "time_utc", "pid", "state", "uptime_sec", "processed_this_run",
        "num_input_files", "num_pending", "num_complete", "num_bad",
        "rss_mb", "cgroup_mb", "cgroup_limit_mb",
        "cuda_allocated_mb", "cuda_reserved_mb",
        "worker_status", "max_files", "max_runtime_sec",
    ]
    if os.path.exists(path):
        with open(path, newline="") as fp:
            existing_header = next(csv.reader(fp), [])
        if existing_header != fields:
            legacy_path = os.path.join(
                cfg.out_monitoring_dir, "realtime_heartbeat_legacy.csv"
            )
            if not os.path.exists(legacy_path):
                shutil.copyfile(path, legacy_path)
            os.remove(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(fields)
        cuda_allocated_mb = 0.0
        cuda_reserved_mb = 0.0
        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                cuda_allocated_mb += torch.cuda.memory_allocated(device_index)
                cuda_reserved_mb += torch.cuda.memory_reserved(device_index)
            cuda_allocated_mb /= 1024.0 ** 2
            cuda_reserved_mb /= 1024.0 ** 2
        cgroup_mb, cgroup_limit_mb = cgroup_memory_mb()
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            os.getpid(), state, round(time.time() - start_time, 1), processed,
            num_input_files, num_pending, num_complete, num_bad,
            round(process_rss_mb(), 2), round(cgroup_mb, 2),
            round(cgroup_limit_mb, 2), round(cuda_allocated_mb, 2),
            round(cuda_reserved_mb, 2), worker_status,
            cfg.max_files, cfg.max_runtime_sec,
        ])
        fp.flush()
        os.fsync(fp.fileno())
    return path

def realtime_loop(
    pickers, subnet_associators, pick_sta_dict, cfg,
    event_repick_coordinator=None, inference_executors=None,
):
    """Poll IN forever-ish, bounded by max_files and max_runtime_sec."""
    start = time.time()
    processed = 0
    done_records = load_done_records(cfg.done_record_path)
    bad_records = load_bad_records(cfg.bad_record_path)
    initial_files = sorted(glob.glob(os.path.join(cfg.in_dir, cfg.in_glob)))
    initial_records = {os.path.abspath(path) for path in initial_files}
    current_done_records = {
        os.path.abspath(path)
        for path in initial_files
        if pipeline_outputs_complete(path, cfg)
    }
    stale_done_records = (
        done_records & initial_records
    ) - current_done_records
    unrecorded_complete = current_done_records - done_records
    skip_records = current_done_records | bad_records
    print("found {} MiniSEED files in IN".format(len(initial_files)))
    print("loaded {} historical completed file records".format(len(done_records)))
    print("validated {} inputs with all current picker/associator outputs".format(
        len(current_done_records)
    ))
    if stale_done_records:
        print("{} historical records are missing current outputs and will be reprocessed".format(
            len(stale_done_records)
        ))
    if unrecorded_complete:
        print("{} inputs have complete outputs without a done-list record; skipping them".format(
            len(unrecorded_complete)
        ))
    print("loaded {} bad file records".format(len(bad_records)))
    print(
        "enabled continuous picker groups: {} | references: {}".format(
            cfg.continuous_picker_groups, cfg.picker_ref_group
        )
    )
    last_wait_report = 0.0
    last_heartbeat = 0.0
    print("realtime limits: max_files={} max_runtime_sec={} (0 means unlimited)".format(
        cfg.max_files, cfg.max_runtime_sec
    ), flush=True)
    if event_repick_coordinator is not None:
        # Startup supplementation may have registered a historical segment.
        # It has completed before the polling loop begins and must not remain
        # resident while the process waits for fresh input.
        event_repick_coordinator.release_segment_cache("before polling")
    print(
        "realtime polling baseline RSS: {:.1f} MB".format(
            release_transient_memory()
        ),
        flush=True,
    )

    while True:
        subnet_associators.assert_healthy()
        all_files = sorted(glob.glob(os.path.join(cfg.in_dir, cfg.in_glob)))
        current_input_records = {os.path.abspath(path) for path in all_files}
        files = []
        for path in all_files:
            record = os.path.abspath(path)
            if record in skip_records:
                continue
            # Output files are authoritative. This also recognizes products
            # restored or generated outside the current Python process.
            if pipeline_outputs_complete(path, cfg):
                current_done_records.add(record)
                skip_records.add(record)
                continue
            files.append(path)
        current_unresolved_bad = (
            bad_records & current_input_records
        ) - current_done_records
        now = time.time()
        if now - last_heartbeat >= 60.0:
            write_realtime_heartbeat(
                cfg, "polling", start, processed, len(all_files), len(files),
                len(current_done_records), len(current_unresolved_bad),
                subnet_associators.health_summary(),
            )
            last_heartbeat = now
        if not files and time.time() - last_wait_report >= 60.0:
            print(
                "waiting for input: {} files present, {} complete for current "
                "pipeline, {} marked bad; polling every {}s".format(
                    len(all_files), len(current_done_records),
                    len(current_unresolved_bad),
                    cfg.poll_interval_sec,
                ),
                flush=True,
            )
            last_wait_report = time.time()
        elif files:
            print("found {} pending MiniSEED files".format(len(files)), flush=True)

        for mseed_path in files:
            if not os.path.exists(mseed_path):
                continue
            print("=" * 60)
            print("processing {}".format(mseed_path))
            segment_error = None
            try:
                pick_paths, phase_paths, picks_by_picker, timing = process_segment(
                    mseed_path,
                    pickers,
                    subnet_associators,
                    pick_sta_dict,
                    cfg,
                    event_repick_coordinator=event_repick_coordinator,
                    inference_executors=inference_executors,
                )
            except Exception as exc:
                segment_error = exc
            finally:
                # Realtime repicking is segment-local. Once association and
                # repicking return, retaining the final hour of all-station
                # filtered data only raises idle RSS and can trigger SIGKILL.
                if event_repick_coordinator is not None:
                    event_repick_coordinator.release_segment_cache(
                        "after {}".format(os.path.basename(mseed_path))
                    )
                if inference_executors is not None:
                    inference_executors.recycle()
                cleanup_rss_mb = release_transient_memory()
                write_segment_memory_stage(
                    cfg.out_monitoring_dir,
                    segment_code(mseed_path),
                    "after_segment_cleanup",
                )

            if segment_error is not None:
                # A dead association worker is a pipeline failure, not bad input.
                subnet_associators.assert_healthy()
                if not isinstance(segment_error, BadRealtimeInputError):
                    print(
                        "pipeline error for {}; input remains retryable: {}: {}"
                        .format(
                            mseed_path,
                            segment_error.__class__.__name__,
                            segment_error,
                        ),
                        flush=True,
                    )
                    raise segment_error.with_traceback(
                        segment_error.__traceback__
                    )
                bad_record = append_bad_record(
                    cfg.bad_record_path, mseed_path, segment_error
                )
                bad_records.add(bad_record)
                skip_records.add(bad_record)
                processed += 1
                print("warning: skip bad realtime file {} | {}: {}".format(
                    mseed_path,
                    segment_error.__class__.__name__,
                    segment_error,
                ))
                if cfg.max_files and processed >= cfg.max_files:
                    print("stop: processed {} files".format(processed))
                    return
                continue

            timing["rss_after_segment_cleanup_mb"] = cleanup_rss_mb
            timing_csv = write_timing_report(
                cfg.out_monitoring_dir, timing["segment"], timing
            )
            print_timing_report(timing)
            done_record = append_done_record(cfg.done_record_path, mseed_path)
            done_records.add(done_record)
            current_done_records.add(done_record)
            skip_records.add(done_record)
            plot_timing_report(
                cfg.timing_plot_script, timing_csv, cfg.out_monitoring_dir
            )
            processed += 1
            for picker_name in pickers:
                print("picker result {}: {} picks | {}".format(
                    picker_name,
                    len(picks_by_picker[picker_name]),
                    pick_paths[picker_name],
                ))
            for branch_name in cfg.association_branches:
                branch = cfg.result_branches[branch_name]
                branch_pick_path = os.path.join(
                    branch["pick_dir"], "{}.pick".format(timing["segment"])
                )
                print("association result {}: {} | {}".format(
                    branch_name,
                    branch_pick_path,
                    phase_paths[branch_name],
                ))
            print("recorded completed input: {}".format(done_record))

            # Do not let the polling loop keep the last segment's result
            # arrays alive through an arbitrary idle period.
            del pick_paths, phase_paths, picks_by_picker, timing
            idle_rss_mb = release_transient_memory()
            print(
                "segment memory released; idle RSS {:.1f} MB".format(
                    idle_rss_mb
                ),
                flush=True,
            )

            if cfg.max_files and processed >= cfg.max_files:
                print("stop: processed {} files".format(processed))
                return
            if cfg.max_runtime_sec and time.time() - start >= cfg.max_runtime_sec:
                print("stop: runtime limit reached")
                return

        if cfg.max_runtime_sec and time.time() - start >= cfg.max_runtime_sec:
            print("stop: runtime limit reached")
            return
        time.sleep(cfg.poll_interval_sec)
