"""Audit event origin times against nominal realtime phase-file windows.

Realtime phase filenames encode the nominal window end. With half-window
overlap, the nominal window duration is twice the median endpoint stride.
"""
import csv
import glob
import os
import re
from datetime import datetime, timedelta
from statistics import median


# ------------------------- User settings -------------------------
INPUT_DIR = "/app/aqms/ai_pal/OUT/phase"
INPUT_PATTERN = "phase_*.dat"

OUT_WARNING_CSV = "/app/aqms/ai_pal/OUT/phase_time_range_warnings.csv"
OUT_SUMMARY_CSV = "/app/aqms/ai_pal/OUT/phase_time_range_summary.csv"

# OT may precede the waveform because OT < TP. Warn only when the origin time
# is farther outside the nominal window than this tolerance.
OUTSIDE_TOL_SEC = 10.0

# Keep 0 to infer duration as 2 x median filename-endpoint stride. Set a
# positive value only when auditing one file or overriding that geometry.
WINDOW_DURATION_SEC = 0

# Near-identical endpoint timestamps are clustered when estimating stride.
ENDPOINT_CLUSTER_TOL_SEC = 5.0
# -----------------------------------------------------------------


EVENT_FIELDS = [
    "source", "event_index", "origin_time", "nominal_start", "nominal_end",
    "status", "outside_sec", "lat", "lon", "depth", "magnitude",
]
SUMMARY_FIELDS = [
    "source", "nominal_start", "nominal_end", "num_events", "num_inside",
    "num_before_within_tolerance", "num_after_within_tolerance",
    "num_warning_before", "num_warning_after", "earliest_origin_time",
    "latest_origin_time", "max_before_sec", "max_after_sec",
]


def parse_time(value):
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def format_time(value):
    if value is None:
        return ""
    text = value.isoformat(timespec="milliseconds")
    return text[:-1] + "Z"


def endpoint_from_path(path):
    match = re.search(r"(\d{8}T\d{6})Z?", os.path.basename(path))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def clustered_endpoints(paths):
    endpoints = sorted({
        endpoint for endpoint in (endpoint_from_path(path) for path in paths)
        if endpoint is not None
    })
    groups = []
    for endpoint in endpoints:
        if (
            not groups
            or (endpoint - groups[-1][-1]).total_seconds()
            > ENDPOINT_CLUSTER_TOL_SEC
        ):
            groups.append([])
        groups[-1].append(endpoint)

    representatives = []
    for group in groups:
        base = group[0]
        offset_sec = median([(item - base).total_seconds() for item in group])
        representatives.append(base + timedelta(seconds=offset_sec))
    return representatives


def infer_window_duration(paths):
    if WINDOW_DURATION_SEC > 0:
        return float(WINDOW_DURATION_SEC), None

    endpoints = clustered_endpoints(paths)
    strides = [
        (endpoints[idx] - endpoints[idx - 1]).total_seconds()
        for idx in range(1, len(endpoints))
    ]
    strides = [stride for stride in strides if stride > 0]
    if not strides:
        raise ValueError(
            "Cannot infer window duration from fewer than two endpoint times; "
            "set WINDOW_DURATION_SEC explicitly"
        )
    stride_sec = float(median(strides))
    return 2.0 * stride_sec, stride_sec


def is_event_header(codes):
    if len(codes) != 5 or "T" not in codes[0]:
        return False
    try:
        parse_time(codes[0])
        for value in codes[1:5]:
            float(value)
    except ValueError:
        return False
    return True


def read_event_headers(path):
    events = []
    with open(path) as fp:
        for line in fp:
            codes = [code.strip() for code in line.split(",")]
            if not is_event_header(codes):
                continue
            events.append({
                "time": parse_time(codes[0]),
                "lat": float(codes[1]),
                "lon": float(codes[2]),
                "depth": float(codes[3]),
                "magnitude": float(codes[4]),
            })
    return events


def classify_origin_time(origin_time, nominal_start, nominal_end):
    if origin_time < nominal_start:
        outside_sec = (nominal_start - origin_time).total_seconds()
        status = (
            "warning_before"
            if outside_sec > OUTSIDE_TOL_SEC
            else "before_within_tolerance"
        )
        return status, outside_sec
    if origin_time > nominal_end:
        outside_sec = (origin_time - nominal_end).total_seconds()
        status = (
            "warning_after"
            if outside_sec > OUTSIDE_TOL_SEC
            else "after_within_tolerance"
        )
        return status, outside_sec
    return "inside", 0.0


def audit_file(path, window_duration_sec):
    nominal_end = endpoint_from_path(path)
    if nominal_end is None:
        raise ValueError("No endpoint timestamp in filename: {}".format(path))
    nominal_start = nominal_end - timedelta(seconds=window_duration_sec)
    events = read_event_headers(path)

    counts = {
        "inside": 0,
        "before_within_tolerance": 0,
        "after_within_tolerance": 0,
        "warning_before": 0,
        "warning_after": 0,
    }
    warnings = []
    max_before_sec = 0.0
    max_after_sec = 0.0
    for event_index, event in enumerate(events):
        status, outside_sec = classify_origin_time(
            event["time"], nominal_start, nominal_end
        )
        counts[status] += 1
        if status in ("before_within_tolerance", "warning_before"):
            max_before_sec = max(max_before_sec, outside_sec)
        if status in ("after_within_tolerance", "warning_after"):
            max_after_sec = max(max_after_sec, outside_sec)
        if status.startswith("warning_"):
            warnings.append({
                "source": path,
                "event_index": event_index,
                "origin_time": format_time(event["time"]),
                "nominal_start": format_time(nominal_start),
                "nominal_end": format_time(nominal_end),
                "status": status,
                "outside_sec": "{:.3f}".format(outside_sec),
                "lat": "{:.5f}".format(event["lat"]),
                "lon": "{:.5f}".format(event["lon"]),
                "depth": "{:.1f}".format(event["depth"]),
                "magnitude": "{:.2f}".format(event["magnitude"]),
            })

    event_times = [event["time"] for event in events]
    summary = {
        "source": path,
        "nominal_start": format_time(nominal_start),
        "nominal_end": format_time(nominal_end),
        "num_events": len(events),
        "num_inside": counts["inside"],
        "num_before_within_tolerance": counts["before_within_tolerance"],
        "num_after_within_tolerance": counts["after_within_tolerance"],
        "num_warning_before": counts["warning_before"],
        "num_warning_after": counts["warning_after"],
        "earliest_origin_time": format_time(min(event_times)) if event_times else "",
        "latest_origin_time": format_time(max(event_times)) if event_times else "",
        "max_before_sec": "{:.3f}".format(max_before_sec),
        "max_after_sec": "{:.3f}".format(max_after_sec),
    }
    return summary, warnings


def write_csv(path, fields, rows):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_PATTERN)))
    if not paths:
        raise SystemExit(
            "No original phase files found: {}".format(
                os.path.join(INPUT_DIR, INPUT_PATTERN)
            )
        )

    window_duration_sec, stride_sec = infer_window_duration(paths)
    if stride_sec is None:
        print("using configured phase window duration: {:.2f}s".format(
            window_duration_sec
        ))
    else:
        print(
            "inferred endpoint stride: {:.2f}s | nominal phase window: {:.2f}s".format(
                stride_sec, window_duration_sec
            )
        )

    summaries = []
    warnings = []
    skipped = 0
    for path in paths:
        try:
            summary, warnings_i = audit_file(path, window_duration_sec)
        except Exception as exc:
            skipped += 1
            print("warning: skip {} | {}: {}".format(
                path, exc.__class__.__name__, exc
            ))
            continue
        summaries.append(summary)
        warnings.extend(warnings_i)
        if warnings_i:
            print("{}: {} events | {} warnings".format(
                path, summary["num_events"], len(warnings_i)
            ))

    write_csv(OUT_WARNING_CSV, EVENT_FIELDS, warnings)
    write_csv(OUT_SUMMARY_CSV, SUMMARY_FIELDS, summaries)
    print("-" * 60)
    print("phase files: {} | skipped: {}".format(len(paths), skipped))
    print("events: {}".format(sum(row["num_events"] for row in summaries)))
    print("warnings: {}".format(len(warnings)))
    print("warning CSV: {}".format(OUT_WARNING_CSV))
    print("summary CSV: {}".format(OUT_SUMMARY_CSV))


if __name__ == "__main__":
    main()