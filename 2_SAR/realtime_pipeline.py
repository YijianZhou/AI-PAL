"""Pseudo-realtime data pipeline for multi-model picking and PAL association.

This module keeps the offline SAR picker and PAL associator unchanged. It adapts
the data ingest/output layer for hourly all-network miniSEED files.
"""
import csv
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from statistics import median

import numpy as np
import torch
from obspy import read, Stream, UTCDateTime


PICK_DTYPE = [
    ("net_sta", "O"),
    ("sta_ot", "O"),
    ("tp", "O"),
    ("ts", "O"),
    ("s_amp", "O"),
    ("p_prob", "O"),
    ("s_prob", "O"),
]


def prepare_and_pick_station(pickers, st_all, sta_key, sta_info, location_priority):
    """Prepare one station once, then run every enabled picker."""
    t0 = time.perf_counter()
    st = traces_for_station(st_all, sta_key, location_priority)
    if len(st) != 0:
        st = normalize_station_stream(st, sta_key, sta_info)
    preprocess_sec = time.perf_counter() - t0
    if len(st) != 3:
        return sta_key, {}, preprocess_sec, {}, False

    print("-" * 40)
    print("picking {} with {}".format(sta_key, ", ".join(pickers)))
    picker_results = {}
    picker_seconds = {}
    for picker_name, picker_i in pickers.items():
        t0 = time.perf_counter()
        with torch.inference_mode():
            picks_i = picker_i.pick(st.copy())
        picker_results[picker_name] = picks_i if picks_i else []
        picker_seconds[picker_name] = time.perf_counter() - t0
    return sta_key, picker_results, preprocess_sec, picker_seconds, True

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

def unpack_picker_pick(pick):
    """Return tp, ts, s_amp, p_prob, s_prob from SAR picker output."""
    if hasattr(pick, "dtype") and getattr(pick.dtype, "names", None):
        names = pick.dtype.names
        tp = pick["tp"]
        ts = pick["ts"]
        s_amp = pick["s_amp"] if "s_amp" in names else -1
        p_prob = pick["p_prob"] if "p_prob" in names else -1
        s_prob = pick["s_prob"] if "s_prob" in names else -1
        return tp, ts, s_amp, p_prob, s_prob

    tp = pick[1]
    ts = pick[2]
    s_amp = pick[3]
    p_prob = pick[4] if len(pick) >= 6 else -1
    s_prob = pick[5] if len(pick) >= 6 else -1
    return tp, ts, s_amp, p_prob, s_prob

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
    st = read(mseed_path)
    timing["data_read_sec"] = time.perf_counter() - t0
    timing["num_raw_traces"] = len(st)

    t0 = time.perf_counter()
    st, dropped = cleanup_sampling_rates(
        st,
        expected_sampling_rate=expected_sampling_rate,
    )
    timing["sampling_rate_cleanup_sec"] = time.perf_counter() - t0
    timing["num_bad_sampling_rate_traces"] = len(dropped)

    t0 = time.perf_counter()
    st.merge(fill_value=0)
    timing["data_merge_sec"] = time.perf_counter() - t0
    timing["num_merged_traces"] = len(st)
    if len(st):
        timing["data_start"] = format_time(
            min(tr.stats.starttime for tr in st).datetime
        )
        timing["data_end"] = format_time(
            max(tr.stats.endtime + tr.stats.delta for tr in st).datetime
        )
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


def select_gain(gain, stream):
    """Select PAL-format gain for a stream midpoint."""
    if isinstance(gain, float):
        return gain
    if isinstance(gain, int):
        return float(gain)
    if not gain:
        return 1.0
    if isinstance(gain[0], float):
        return gain
    if isinstance(gain[0], int):
        return [float(item) for item in gain]

    start_time = max([tr.stats.starttime for tr in stream])
    end_time = min([tr.stats.endtime for tr in stream])
    st_time = start_time + (end_time - start_time) / 2
    selected = gain[0][0:3]
    for gain_row in gain:
        t0, t1 = UTCDateTime(gain_row[3]), UTCDateTime(gain_row[4])
        if t0 < st_time < t1:
            selected = gain_row[0:3]
            break
    return selected


def apply_gain_and_units(stream, sta_key, sta_info):
    """Convert counts to velocity units expected by SAR/PAL magnitude."""
    gain = select_gain(sta_info[3], stream)
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
    """Convert one station/location group into E,N,Z order for AI-PAL."""
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
    elif len(st) == 1:
        out = Stream([st[0].copy(), st[0].copy(), st[0].copy()])
    else:
        print("skip {}: expected 1 or 3 usable channels, got {}".format(sta_key, len(st)))
        return Stream()

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
                tp, ts, s_amp, p_prob, s_prob = unpack_picker_pick(pick)
                sta_ot = calc_ot(tp, ts, vp=vp, vs=vs)
                picks.append((sta_key, sta_ot, tp, ts, s_amp, p_prob, s_prob))
                fout.write(
                    "{},{},{},{},{:.4f},{:.4f}\n".format(
                        sta_key, format_time(tp), format_time(ts), s_amp, p_prob, s_prob
                    )
                )

    return np.array(picks, dtype=PICK_DTYPE), pick_path


def _timing_key(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def pick_segment_timed(mseed_path, pickers, sta_dict, pick_dirs,
                       location_priority=None, vp=6.0, vs=3.45,
                       num_workers=0, expected_sampling_rate=None):
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
    }
    pick_paths = {}
    outputs = {}
    picks_by_picker = {name: [] for name in pickers}
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
    station_keys = sorted(sta_dict)
    picking_wall_t0 = time.perf_counter()
    try:
        if num_workers and num_workers > 1:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            result_iter = executor.map(
                lambda sta_key: prepare_and_pick_station(
                    pickers, st_all, sta_key, sta_dict[sta_key], location_priority
                ),
                station_keys,
            )
        else:
            executor = None
            result_iter = (
                prepare_and_pick_station(
                    pickers, st_all, sta_key, sta_dict[sta_key], location_priority
                )
                for sta_key in station_keys
            )

        for sta_key, station_results, prep_sec, picker_seconds, has_stream in result_iter:
            timing["preprocess_sec"] += float(prep_sec)
            if has_stream:
                timing["num_station_streams"] += 1
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
                    tp, ts, s_amp, p_prob, s_prob = unpack_picker_pick(pick)
                    sta_ot = calc_ot(tp, ts, vp=vp, vs=vs)
                    picks_by_picker[picker_name].append(
                        (sta_key, sta_ot, tp, ts, s_amp, p_prob, s_prob)
                    )
                    outputs[picker_name].write(
                        "{},{},{},{},{:.4f},{:.4f}\n".format(
                            sta_key, format_time(tp), format_time(ts),
                            s_amp, p_prob, s_prob,
                        )
                    )
                timing["pick_write_sec"] += time.perf_counter() - t0
        if executor is not None:
            executor.shutdown()
    finally:
        for output in outputs.values():
            output.close()
    timing["picking_sec"] = time.perf_counter() - picking_wall_t0
    del st_all

    arrays = {}
    for picker_name, picks in picks_by_picker.items():
        arrays[picker_name] = np.array(picks, dtype=PICK_DTYPE)
        timing["num_picks_{}".format(_timing_key(picker_name))] = len(picks)
    timing["num_picks"] = sum(len(picks) for picks in arrays.values())
    return arrays, pick_paths, timing


def process_segment(mseed_path, pickers, subnet_associators, pick_sta_dict, cfg):
    timing = {"segment": segment_code(mseed_path)}
    segment_t0 = time.perf_counter()
    pick_dirs = {
        name: cfg.result_branches[name]["pick_dir"] for name in pickers
    }
    picks_by_picker, pick_paths, pick_timing = pick_segment_timed(
        mseed_path,
        pickers,
        pick_sta_dict,
        pick_dirs,
        location_priority=cfg.location_priority,
        vp=cfg.vp,
        vs=cfg.vs,
        num_workers=cfg.num_workers,
        expected_sampling_rate=cfg.samp_rate,
    )
    timing.update(pick_timing)

    segment = segment_code(mseed_path)
    phase_paths = {}
    timing["assoc_sec"] = 0.0
    timing["assoc_wall_sec"] = 0.0
    timing["merge_sec"] = 0.0
    timing["time_segment_merge_sec"] = 0.0
    timing["num_final_intervals_written"] = 0
    timing["num_final_events_written"] = 0

    for picker_name, picks in picks_by_picker.items():
        branch = cfg.result_branches[picker_name]
        branch_key = _timing_key(picker_name)
        for out_dir in (
            branch["phase_dir"], branch["subnet_phase_dir"],
            branch["merge_dir"], branch["final_cfg"].out_final_pha_dir,
            branch["final_cfg"].out_final_merge_dir, cfg.out_timing_dir,
        ):
            if not os.path.exists(out_dir):
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
        phase_paths[picker_name] = merged_phase_path

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
            print("warning: final time merge failed for {} {} | {}: {}".format(
                picker_name, segment, exc.__class__.__name__, exc
            ))
        final_sec = time.perf_counter() - t0
        timing["time_segment_merge_{}_sec".format(branch_key)] = final_sec
        timing["time_segment_merge_sec"] += final_sec
        timing["num_final_intervals_written"] += len(final_results)
        timing["num_final_events_written"] += sum(
            result["num_merged_events"] for result in final_results
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
            picks_by_station = {}
            for pick in event["picks"]:
                picks_by_station.setdefault(pick["sta"], pick)
            for station, pick in picks_by_station.items():
                entries_by_station.setdefault(station, []).append(
                    (pick["p"], pick["s"], event_idx)
                )

        matched_station_counts = {}
        for entries in entries_by_station.values():
            entries.sort(key=lambda item: item[0])
            for left_pos, (left_p, left_s, left_idx) in enumerate(entries):
                right_pos = left_pos + 1
                while right_pos < len(entries):
                    right_p, right_s, right_idx = entries[right_pos]
                    p_dt = (right_p - left_p).total_seconds()
                    if p_dt >= phase_pick_time_tol_sec:
                        break
                    if (
                        left_idx != right_idx
                        and abs((right_s - left_s).total_seconds())
                        < phase_pick_time_tol_sec
                    ):
                        pair = tuple(sorted((left_idx, right_idx)))
                        matched_station_counts[pair] = (
                            matched_station_counts.get(pair, 0) + 1
                        )
                    right_pos += 1

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


def merge_group(group):
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
        picks = picks_by_sta[sta]
        merged["picks"].append(
            {
                "sta": sta,
                "p": median_time([pick["p"] for pick in picks]),
                "s": median_time([pick["s"] for pick in picks]),
                "score": median([pick["score"] for pick in picks]),
                "p_prob": median_valid([pick["p_prob"] for pick in picks], default=-1, min_value=0),
                "s_prob": median_valid([pick["s_prob"] for pick in picks], default=-1, min_value=0),
                "num_picks": len(picks),
            }
        )

    return merged


def merge_phase_files(fpha_list, fpha_out, fmerge_log, origin_time_tol_sec=2.5,
                      epicenter_tol_km=5.0, depth_tol_km=10.0,
                      time_format_digits=6, event_time_start=None,
                      event_time_end=None, exclude_phase_files=None,
                      min_shared_phase_stations=0,
                      phase_pick_time_tol_sec=1.0):
    events = []
    file_event_counts = {}
    for fpha in sorted(fpha_list):
        file_events = read_phase_file(fpha)
        file_event_counts[fpha] = len(file_events)
        events.extend(file_events)
        print("{}: {} events".format(fpha, len(file_events)))

    num_input_events = len(events)
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
    if event_time_start is not None or event_time_end is not None:
        groups = [
            group for group in groups
            if any(
                (event_time_start is None or event["time"] >= event_time_start)
                and (event_time_end is None or event["time"] < event_time_end)
                for event in group["events"]
            )
        ]
    merged_events = []
    for group in groups:
        merged = merge_group(group)
        if event_time_start is not None or event_time_end is not None:
            interval_times = [
                event["time"] for event in group["events"]
                if (event_time_start is None or event["time"] >= event_time_start)
                and (event_time_end is None or event["time"] < event_time_end)
            ]
            merged["time"] = median_time(interval_times)
        merged_events.append(merged)
    merged_events = sorted(merged_events, key=lambda item: item["time"])

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
                    "{},{},{},{},{:.4f},{:.4f}\n".format(
                        pick["sta"],
                        format_time(pick["p"], time_format_digits),
                        format_time(pick["s"], time_format_digits),
                        pick["score"],
                        pick["p_prob"],
                        pick["s_prob"],
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
        "num_merged_events": len(merged_events),
        "num_duplicate_events_removed": num_input_events - num_grouped_events,
        "num_events_outside_final_interval": num_grouped_events - len(merged_events),
        "num_multi_input_event_groups": num_multi_event_groups,
        "max_input_events_per_group": max_input_events_per_group,
    }
    print("input phase files: {}".format(summary["num_input_phase_files"]))
    print("input events: {}".format(summary["num_input_events"]))
    print("merged events: {}".format(summary["num_merged_events"]))
    print("duplicate input events removed: {}".format(summary["num_duplicate_events_removed"]))
    print("multi-input-event groups: {}".format(summary["num_multi_input_event_groups"]))
    print("max input events per merged group: {}".format(summary["max_input_events_per_group"]))
    print("output phase file: {}".format(fpha_out))
    print("merge log: {}".format(fmerge_log))
    return summary


TIME_SEGMENT_MERGE_VERSION = "6"
SEGMENT_WINDOW_FIELDS = [
    "segment", "start", "end", "phase_path", "bounds_source",
]
FINALIZED_WINDOW_FIELDS = [
    "merge_version", "start", "end", "phase_path", "catalog_path",
    "merge_log_path", "previous_segment", "current_segment", "num_events",
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


def _waveform_bounds(mseed_path):
    """Read miniSEED headers only and return the represented time interval."""
    st = read(mseed_path, headonly=True)
    if len(st) == 0:
        return None
    start = min(tr.stats.starttime for tr in st).datetime
    end = max(tr.stats.endtime + tr.stats.delta for tr in st).datetime
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

def _publish_final_interval(previous, current, interval_start, interval_end, cfg,
                            exclude_phase_path=None):
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
        exclude_phase_files=[exclude_phase_path] if exclude_phase_path else None,
    )
    if write_catalog:
        write_catalog_from_phase(phase_tmp, catalog_tmp)

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
    if len(current_rows) != len(all_rows):
        _remove_obsolete_final_outputs(cfg)
        current_rows = []
    _write_csv_atomic(
        cfg.finalized_window_path,
        FINALIZED_WINDOW_FIELDS,
        current_rows,
    )
    return current_rows


def _finalize_segment_windows(records, cfg):
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
    last_final_phase_path = None

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
            last_final_phase_path = None
            continue

        # With half-window overlap, the finalized chunk is one arrival stride
        # ending at the earlier source-window endpoint. These chunks are
        # disjoint even when waveform coverage or old metadata is irregular.
        interval_end = previous["end"]
        interval_start = interval_end - timedelta(seconds=stride_sec)
        pair = (previous["id"], current["id"])

        if pair in finalized_pairs:
            last_final_phase_path = finalized_by_pair[pair].get("phase_path")
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
            exclude_phase_path=last_final_phase_path,
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
        last_final_phase_path = summary["phase_path"]
        results.append(summary)

    return results

def update_final_merged_outputs(segment, phase_path, data_start, data_end, cfg):
    """Register a new phase window and publish newly finalized intervals."""
    records = _register_segment_window(
        segment, phase_path, data_start, data_end, cfg
    )
    return _finalize_segment_windows(records, cfg)


def bootstrap_final_merged_outputs(cfg):
    """Finalize all pending original phase files before realtime polling."""
    t0 = time.perf_counter()
    out_dirs = [
        cfg.out_final_pha_dir,
        cfg.out_final_merge_dir,
        cfg.out_timing_dir,
    ]
    if getattr(cfg, "write_catalog_outputs", True):
        out_dirs.append(cfg.out_final_ctlg_dir)
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    num_phase_files = len(_phase_files_by_segment(cfg))
    print(
        "startup final merge: found {} original phase files in {}".format(
            num_phase_files, cfg.out_pha_dir
        ),
        flush=True,
    )
    # Historical startup must be fast. Infer the half-overlap window geometry
    # from phase filename timestamps; do not parse large all-station miniSEEDs.
    records = _discover_existing_segment_windows(cfg, refresh_waveforms=False)
    results = _finalize_segment_windows(records, cfg)
    elapsed = time.perf_counter() - t0
    num_events = sum(result["num_merged_events"] for result in results)
    print(
        "startup final time merge: {:.2f}s | {} phase windows | "
        "{} final intervals written".format(
            elapsed, len(records), len(results)
        ),
        flush=True,
    )

    if not os.path.exists(cfg.out_timing_dir):
        os.makedirs(cfg.out_timing_dir)
    timing_path = os.path.join(
        cfg.out_timing_dir, "timing_final_merge_startup_multi_picker.csv"
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


def pipeline_outputs_complete(mseed_path, cfg):
    """Return true only when every enabled picker branch has its outputs."""
    segment = segment_code(mseed_path)
    for picker_name in cfg.enabled_pickers:
        branch = cfg.result_branches[picker_name]
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
        if key.endswith("_sec") or key.startswith("num_"):
            print("  {}: {}".format(key, timing[key]))


def realtime_loop(pickers, subnet_associators, pick_sta_dict, cfg):
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
    print("enabled picker groups: {}".format(cfg.picker_groups))
    last_wait_report = 0.0

    while True:
        all_files = sorted(glob.glob(os.path.join(cfg.in_dir, cfg.in_glob)))
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
        if not files and time.time() - last_wait_report >= 60.0:
            print(
                "waiting for input: {} files present, {} complete for current "
                "pipeline, {} marked bad; polling every {}s".format(
                    len(all_files), len(current_done_records), len(bad_records),
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
            try:
                pick_paths, phase_paths, picks_by_picker, timing = process_segment(
                    mseed_path, pickers, subnet_associators, pick_sta_dict, cfg
                )
            except Exception as exc:
                bad_record = append_bad_record(cfg.bad_record_path, mseed_path, exc)
                bad_records.add(bad_record)
                skip_records.add(bad_record)
                processed += 1
                print("warning: skip bad realtime file {} | {}: {}".format(
                    mseed_path, exc.__class__.__name__, exc
                ))
                if cfg.max_files and processed >= cfg.max_files:
                    print("stop: processed {} files".format(processed))
                    return
                continue

            timing_csv = write_timing_report(
                cfg.out_timing_dir, timing["segment"], timing
            )
            print_timing_report(timing)
            done_record = append_done_record(cfg.done_record_path, mseed_path)
            done_records.add(done_record)
            current_done_records.add(done_record)
            skip_records.add(done_record)
            plot_timing_report(
                cfg.timing_plot_script, timing_csv, cfg.out_timing_dir
            )
            processed += 1
            for picker_name in pickers:
                print("segment result {}: {} picks | {} | {}".format(
                    picker_name,
                    len(picks_by_picker[picker_name]),
                    pick_paths[picker_name],
                    phase_paths[picker_name],
                ))
            print("recorded completed input: {}".format(done_record))

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