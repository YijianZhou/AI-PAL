"""Audit miniSEED trace coverage against filename-defined nominal windows.

SCSN realtime filenames encode the nominal window end. With half-window
overlap, nominal window duration is twice the median filename-endpoint stride.
"""
import csv
import glob
import os
import re
import time
from datetime import datetime, timedelta
from statistics import median

from obspy import read


# ------------------------- User settings -------------------------
INPUT_DIR = "/app/aqms/ai_pal/IN"
INPUT_PATTERN = "*.ms"

OUT_WARNING_CSV = "/app/aqms/ai_pal/OUT/mseed_time_range_warnings.csv"
OUT_SUMMARY_CSV = "/app/aqms/ai_pal/OUT/mseed_time_range_summary.csv"

# Warn when a trace extends farther than this beyond either nominal boundary.
OUTSIDE_TOL_SEC = 10.0

# Keep 0 to infer duration as 2 x median filename-endpoint stride. Set a
# positive value when checking one file or overriding the inferred geometry.
WINDOW_DURATION_SEC = 0
ENDPOINT_CLUSTER_TOL_SEC = 5.0
# -----------------------------------------------------------------


WARNING_FIELDS = [
    "source", "trace_id", "nominal_start", "nominal_end", "trace_start",
    "trace_end_exclusive", "status", "before_sec", "after_sec",
    "sampling_rate", "npts", "error_type", "error",
]
SUMMARY_FIELDS = [
    "source", "nominal_start", "nominal_end", "num_traces",
    "num_warning_traces", "earliest_trace_start", "latest_trace_end_exclusive",
    "max_before_sec", "max_after_sec", "read_sec", "status", "error",
]


def endpoint_from_path(path):
    match = re.search(r"(\d{8}T\d{6})Z?", os.path.basename(path))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def format_time(value):
    if value is None:
        return ""
    if hasattr(value, "datetime"):
        value = value.datetime
    text = value.isoformat(timespec="milliseconds")
    return text[:-1] + "Z"


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
            "Cannot infer window duration from fewer than two filenames; "
            "set WINDOW_DURATION_SEC explicitly"
        )
    stride_sec = float(median(strides))
    return 2.0 * stride_sec, stride_sec


def warning_status(before_sec, after_sec):
    before = before_sec > OUTSIDE_TOL_SEC
    after = after_sec > OUTSIDE_TOL_SEC
    if before and after:
        return "warning_before_and_after"
    if before:
        return "warning_before"
    if after:
        return "warning_after"
    return None


def audit_file(path, window_duration_sec):
    nominal_end = endpoint_from_path(path)
    if nominal_end is None:
        raise ValueError("No endpoint timestamp in filename: {}".format(path))
    nominal_start = nominal_end - timedelta(seconds=window_duration_sec)

    t0 = time.perf_counter()
    stream = read(path, headonly=True)
    read_sec = time.perf_counter() - t0
    warnings = []
    starts = []
    ends = []
    max_before_sec = 0.0
    max_after_sec = 0.0

    for trace in stream:
        trace_start = trace.stats.starttime.datetime
        trace_end = (trace.stats.endtime + trace.stats.delta).datetime
        starts.append(trace_start)
        ends.append(trace_end)
        before_sec = max(0.0, (nominal_start - trace_start).total_seconds())
        after_sec = max(0.0, (trace_end - nominal_end).total_seconds())
        max_before_sec = max(max_before_sec, before_sec)
        max_after_sec = max(max_after_sec, after_sec)
        status = warning_status(before_sec, after_sec)
        if status is None:
            continue
        warnings.append({
            "source": path,
            "trace_id": trace.id,
            "nominal_start": format_time(nominal_start),
            "nominal_end": format_time(nominal_end),
            "trace_start": format_time(trace_start),
            "trace_end_exclusive": format_time(trace_end),
            "status": status,
            "before_sec": "{:.3f}".format(before_sec),
            "after_sec": "{:.3f}".format(after_sec),
            "sampling_rate": trace.stats.sampling_rate,
            "npts": trace.stats.npts,
            "error_type": "",
            "error": "",
        })

    summary = {
        "source": path,
        "nominal_start": format_time(nominal_start),
        "nominal_end": format_time(nominal_end),
        "num_traces": len(stream),
        "num_warning_traces": len(warnings),
        "earliest_trace_start": format_time(min(starts)) if starts else "",
        "latest_trace_end_exclusive": format_time(max(ends)) if ends else "",
        "max_before_sec": "{:.3f}".format(max_before_sec),
        "max_after_sec": "{:.3f}".format(max_after_sec),
        "read_sec": "{:.3f}".format(read_sec),
        "status": "ok" if not warnings else "warning",
        "error": "",
    }
    return summary, warnings


def error_rows(path, exc, read_sec):
    endpoint = endpoint_from_path(path)
    warning = {
        "source": path,
        "trace_id": "",
        "nominal_start": "",
        "nominal_end": format_time(endpoint),
        "trace_start": "",
        "trace_end_exclusive": "",
        "status": "read_error",
        "before_sec": "",
        "after_sec": "",
        "sampling_rate": "",
        "npts": "",
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    summary = {
        "source": path,
        "nominal_start": "",
        "nominal_end": format_time(endpoint),
        "num_traces": 0,
        "num_warning_traces": 0,
        "earliest_trace_start": "",
        "latest_trace_end_exclusive": "",
        "max_before_sec": "",
        "max_after_sec": "",
        "read_sec": "{:.3f}".format(read_sec),
        "status": "read_error",
        "error": "{}: {}".format(exc.__class__.__name__, exc),
    }
    return summary, warning


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
        raise SystemExit("No miniSEED files found: {}".format(
            os.path.join(INPUT_DIR, INPUT_PATTERN)
        ))

    window_duration_sec, stride_sec = infer_window_duration(paths)
    if stride_sec is None:
        print("using configured miniSEED window duration: {:.2f}s".format(
            window_duration_sec
        ))
    else:
        print(
            "inferred endpoint stride: {:.2f}s | nominal miniSEED window: {:.2f}s".format(
                stride_sec, window_duration_sec
            )
        )

    summaries = []
    warnings = []
    for file_index, path in enumerate(paths, start=1):
        print("[{}/{}] checking {}".format(file_index, len(paths), path), flush=True)
        t0 = time.perf_counter()
        try:
            summary, warnings_i = audit_file(path, window_duration_sec)
        except Exception as exc:
            summary, warning = error_rows(path, exc, time.perf_counter() - t0)
            warnings_i = [warning]
            print("warning: {} | {}: {}".format(
                path, exc.__class__.__name__, exc
            ))
        summaries.append(summary)
        warnings.extend(warnings_i)
        if warnings_i and summary["status"] != "read_error":
            print("  {} traces | {} outside-range warnings".format(
                summary["num_traces"], len(warnings_i)
            ))

    write_csv(OUT_WARNING_CSV, WARNING_FIELDS, warnings)
    write_csv(OUT_SUMMARY_CSV, SUMMARY_FIELDS, summaries)
    print("-" * 60)
    print("miniSEED files: {}".format(len(paths)))
    print("trace/read warnings: {}".format(len(warnings)))
    print("warning CSV: {}".format(OUT_WARNING_CSV))
    print("summary CSV: {}".format(OUT_SUMMARY_CSV))


if __name__ == "__main__":
    main()
