"""Check PAL catalog/phase files for potential duplicate events.

This is an offline audit utility for realtime AI-PAL outputs.  It scans one
phase/catalog file, a directory of files, or a glob, then reports event pairs and
connected duplicate groups using origin-time, epicentral-distance, and depth
thresholds.
"""
import csv
import glob
import math
import os
from datetime import datetime


# ------------------------- User settings -------------------------
# Check the public time-segment-merged phase files by default. To inspect the
# original windows, use OUT/phase with pattern "phase_*.dat".
INPUT_PATH = "/app/aqms/ai_pal/OUT/phase_final"
INPUT_PATTERN = "phase_final_*.dat"

OUT_PAIR_CSV = "/app/aqms/ai_pal/OUT/duplicate_event_pairs.csv"
OUT_GROUP_CSV = "/app/aqms/ai_pal/OUT/duplicate_event_groups.csv"

# Duplicate-detection thresholds.  Use the same values as the merge params when
# checking whether merge left candidate duplicates behind.
ORIGIN_TIME_TOL_SEC = 2.5
EPICENTER_TOL_KM = 5.0
DEPTH_TOL_KM = 10.0

# Independent phase-pick duplicate rule: both P and S must differ by less than
# this tolerance at at least this many common stations.
MIN_MATCHED_PHASE_STATIONS = 4
PHASE_PICK_TIME_TOL_SEC = 1.0

# Optional additional check for phase files.  Keep 0 for catalog-only behavior.
MIN_SHARED_STATIONS = 0
# -----------------------------------------------------------------


class Config(object):
    input = INPUT_PATH
    pattern = INPUT_PATTERN
    out_csv = OUT_PAIR_CSV
    out_group_csv = OUT_GROUP_CSV
    origin_time_tol_sec = ORIGIN_TIME_TOL_SEC
    epicenter_tol_km = EPICENTER_TOL_KM
    depth_tol_km = DEPTH_TOL_KM
    min_matched_phase_stations = MIN_MATCHED_PHASE_STATIONS
    phase_pick_time_tol_sec = PHASE_PICK_TIME_TOL_SEC
    min_shared_stations = MIN_SHARED_STATIONS


def input_files(input_path, pattern):
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, pattern)))
    matches = sorted(glob.glob(input_path))
    if matches:
        return matches
    if os.path.isfile(input_path):
        return [input_path]
    return []


def default_output_path(input_path, filename):
    if os.path.isdir(input_path):
        return os.path.join(input_path, filename)
    parent = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(parent or ".", filename)


def parse_time(value):
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def format_time(value):
    text = value.isoformat(timespec="milliseconds")
    return text[:-1] + "Z"


def is_event_header(codes):
    if len(codes) != 5 or "T" not in codes[0]:
        return False
    try:
        parse_time(codes[0])
        float(codes[1])
        float(codes[2])
        float(codes[3])
        float(codes[4])
    except ValueError:
        return False
    return True


def is_pick_row(codes):
    return len(codes) >= 4 and "T" in codes[1] and "T" in codes[2]


def read_events(path):
    events = []
    current = None
    with open(path) as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            codes = [code.strip() for code in line.split(",")]
            if is_event_header(codes):
                if current is not None:
                    events.append(current)
                current = {
                    "source": path,
                    "line_no": line_no,
                    "event_index": len(events),
                    "time": parse_time(codes[0]),
                    "lat": float(codes[1]),
                    "lon": float(codes[2]),
                    "depth": float(codes[3]),
                    "mag": float(codes[4]),
                    "picks": {},
                    "num_picks": 0,
                }
            elif current is not None and is_pick_row(codes):
                current["picks"].setdefault(codes[0], []).append(
                    (parse_time(codes[1]), parse_time(codes[2]))
                )
                current["num_picks"] += 1
            elif current is None:
                # A pure catalog file has event rows only; ignore anything else.
                continue
        if current is not None:
            events.append(current)
    return events


def horizontal_distance_km(left, right):
    lat0 = 0.5 * (left["lat"] + right["lat"])
    dx = (right["lon"] - left["lon"]) * 111.32 * math.cos(math.radians(lat0))
    dy = (right["lat"] - left["lat"]) * 111.32
    return math.hypot(dx, dy)


def matched_phase_stations(left, right, tolerance_sec):
    matched = 0
    for station in set(left["picks"]) & set(right["picks"]):
        station_matches = False
        for left_p, left_s in left["picks"][station]:
            for right_p, right_s in right["picks"][station]:
                if (
                    abs((right_p - left_p).total_seconds()) < tolerance_sec
                    and abs((right_s - left_s).total_seconds()) < tolerance_sec
                ):
                    station_matches = True
                    break
            if station_matches:
                break
        if station_matches:
            matched += 1
    return matched


def event_pair_metrics(left, right, args):
    matched_stations = matched_phase_stations(
        left, right, args.phase_pick_time_tol_sec
    )
    return {
        "dt_sec": abs((right["time"] - left["time"]).total_seconds()),
        "dist_km": horizontal_distance_km(left, right),
        "ddepth_km": abs(right["depth"] - left["depth"]),
        "shared_stations": len(set(left["picks"]) & set(right["picks"])),
        "matched_phase_stations": matched_stations,
    }


def is_duplicate_pair(metrics, args):
    if metrics["matched_phase_stations"] >= args.min_matched_phase_stations:
        return True
    if metrics["dt_sec"] > args.origin_time_tol_sec:
        return False
    if metrics["dist_km"] > args.epicenter_tol_km:
        return False
    if metrics["ddepth_km"] > args.depth_tol_km:
        return False
    if metrics["shared_stations"] < args.min_shared_stations:
        return False
    return True


def phase_link_candidate_pairs(events, args):
    if args.min_matched_phase_stations <= 0:
        return set()

    entries_by_station = {}
    for event_idx, event in enumerate(events):
        for station, picks in event["picks"].items():
            for p_time, s_time in picks:
                entries_by_station.setdefault(station, []).append(
                    (p_time, s_time, event_idx)
                )

    stations_by_pair = {}
    for station, entries in entries_by_station.items():
        entries.sort(key=lambda item: item[0])
        for left_pos, (left_p, left_s, left_idx) in enumerate(entries):
            right_pos = left_pos + 1
            while right_pos < len(entries):
                right_p, right_s, right_idx = entries[right_pos]
                p_dt = (right_p - left_p).total_seconds()
                if p_dt >= args.phase_pick_time_tol_sec:
                    break
                if (
                    left_idx != right_idx
                    and abs((right_s - left_s).total_seconds())
                    < args.phase_pick_time_tol_sec
                ):
                    pair = tuple(sorted((left_idx, right_idx)))
                    stations_by_pair.setdefault(pair, set()).add(station)
                right_pos += 1

    return {
        pair for pair, stations in stations_by_pair.items()
        if len(stations) >= args.min_matched_phase_stations
    }


def find_duplicate_pairs(events, args):
    candidate_pairs = phase_link_candidate_pairs(events, args)
    events_by_time = sorted(enumerate(events), key=lambda item: item[1]["time"])
    for left_pos, (left_idx, left_event) in enumerate(events_by_time):
        right_pos = left_pos + 1
        while right_pos < len(events_by_time):
            right_idx, right_event = events_by_time[right_pos]
            dt = (right_event["time"] - left_event["time"]).total_seconds()
            if dt > args.origin_time_tol_sec:
                break
            candidate_pairs.add(tuple(sorted((left_idx, right_idx))))
            right_pos += 1

    pairs = []
    for left_idx, right_idx in sorted(candidate_pairs):
        metrics = event_pair_metrics(events[left_idx], events[right_idx], args)
        if not is_duplicate_pair(metrics, args):
            continue
        duplicate_rule = (
            "phase_picks"
            if metrics["matched_phase_stations"] >= args.min_matched_phase_stations
            else "origin_location"
        )
        pairs.append({
            "left_idx": left_idx,
            "right_idx": right_idx,
            "duplicate_rule": duplicate_rule,
            **metrics,
        })
    return pairs

def duplicate_groups(events, pairs):
    parent = list(range(len(events)))

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

    for pair in pairs:
        union(pair["left_idx"], pair["right_idx"])

    groups = {}
    for idx in range(len(events)):
        groups.setdefault(find(idx), []).append(idx)
    return [sorted(indices, key=lambda idx: events[idx]["time"]) for indices in groups.values() if len(indices) > 1]


def event_id(event):
    return "{}:{}".format(os.path.basename(event["source"]), event["event_index"])


def write_pair_csv(path, events, pairs):
    fields = [
        "pair_id",
        "duplicate_rule",
        "event_a",
        "event_b",
        "time_a",
        "time_b",
        "lat_a",
        "lon_a",
        "depth_a",
        "mag_a",
        "lat_b",
        "lon_b",
        "depth_b",
        "mag_b",
        "dt_sec",
        "dist_km",
        "ddepth_km",
        "shared_stations",
        "matched_phase_stations",
        "source_a",
        "source_b",
    ]
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for pair_id, pair in enumerate(pairs):
            left = events[pair["left_idx"]]
            right = events[pair["right_idx"]]
            writer.writerow({
                "pair_id": pair_id,
                "duplicate_rule": pair["duplicate_rule"],
                "event_a": event_id(left),
                "event_b": event_id(right),
                "time_a": format_time(left["time"]),
                "time_b": format_time(right["time"]),
                "lat_a": "{:.5f}".format(left["lat"]),
                "lon_a": "{:.5f}".format(left["lon"]),
                "depth_a": "{:.1f}".format(left["depth"]),
                "mag_a": "{:.2f}".format(left["mag"]),
                "lat_b": "{:.5f}".format(right["lat"]),
                "lon_b": "{:.5f}".format(right["lon"]),
                "depth_b": "{:.1f}".format(right["depth"]),
                "mag_b": "{:.2f}".format(right["mag"]),
                "dt_sec": "{:.3f}".format(pair["dt_sec"]),
                "dist_km": "{:.3f}".format(pair["dist_km"]),
                "ddepth_km": "{:.3f}".format(pair["ddepth_km"]),
                "shared_stations": pair["shared_stations"],
                "matched_phase_stations": pair["matched_phase_stations"],
                "source_a": left["source"],
                "source_b": right["source"],
            })


def write_group_csv(path, events, groups):
    fields = [
        "group_id",
        "num_events",
        "event_ids",
        "times",
        "sources",
        "num_total_picks",
        "num_unique_stations",
    ]
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for group_id, indices in enumerate(groups):
            group_events = [events[idx] for idx in indices]
            stations = set()
            for event in group_events:
                stations.update(event["picks"])
            writer.writerow({
                "group_id": group_id,
                "num_events": len(group_events),
                "event_ids": "|".join(event_id(event) for event in group_events),
                "times": "|".join(format_time(event["time"]) for event in group_events),
                "sources": "|".join(event["source"] for event in group_events),
                "num_total_picks": sum(event["num_picks"] for event in group_events),
                "num_unique_stations": len(stations),
            })


def main():
    args = Config()
    files = input_files(args.input, args.pattern)
    if not files:
        raise SystemExit("No input files found: {}".format(args.input))

    events = []
    for path in files:
        file_events = read_events(path)
        events.extend(file_events)
        print("{}: {} events".format(path, len(file_events)))

    pairs = find_duplicate_pairs(events, args)
    groups = duplicate_groups(events, pairs)

    out_csv = args.out_csv or default_output_path(args.input, "duplicate_event_pairs.csv")
    out_group_csv = args.out_group_csv or default_output_path(args.input, "duplicate_event_groups.csv")
    write_pair_csv(out_csv, events, pairs)
    write_group_csv(out_group_csv, events, groups)

    print("-" * 60)
    print("input files: {}".format(len(files)))
    print("input events: {}".format(len(events)))
    print("duplicate pairs: {}".format(len(pairs)))
    print("duplicate groups: {}".format(len(groups)))
    if groups:
        print("max events in one duplicate group: {}".format(max(len(group) for group in groups)))
    print("pair CSV: {}".format(out_csv))
    print("group CSV: {}".format(out_group_csv))


if __name__ == "__main__":
    main()
