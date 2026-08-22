"""Extract CEED HDF5 metadata into a PAL-style phase file.

CEED waveform files are event-grouped HDF5 files. Event groups contain origin
metadata, and each station waveform dataset stores phase-arrival attributes.
This script writes one event header row:

    ot,lat,lon,dep,mag,event_id

followed by station pick rows:

    net.sta.loc.chn,tp,ts,dist_km

The station key preserves the CEED dataset key, for example ``CI.CCC..HH``.
Only arrivals whose per-pick ``event_id`` matches the current HDF5 event group
are used when that attribute is present.
"""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


DEFAULT_CEED_ROOT = Path("/nas/zhouyj/CEED")
MISSING_PICK = "-1"

# =============================================================================
# USER SETTINGS
# =============================================================================
CEED_ROOT = DEFAULT_CEED_ROOT
NC_DIR = None  # Optional explicit quakeflow_nc/waveform_h5 path.
SC_DIR = None  # Optional explicit quakeflow_sc/waveform_h5 path.
YEARS = None  # Example: [2019, 2020]; None processes every available year.
PHASE_OUT = Path("output/ceed_phase.pha")
STATION_COUNTS_OUT = Path("output/ceed_phase_station_counts.csv")
PROGRESS_EVERY = 10000


def scalar_attr(attrs, name, default=None):
    if name not in attrs:
        return default
    value = attrs[name]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    return decode_value(value)


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def list_attr(attrs, name):
    if name not in attrs:
        return []
    value = attrs[name]
    if isinstance(value, np.ndarray):
        return [decode_value(item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [decode_value(item) for item in value]
    return [decode_value(value)]


def parse_time(value):
    if value is None:
        return None
    text = str(decode_value(value)).strip()
    if not text or text in {"-1", "nan", "NaN"}:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_time(value):
    if value is None:
        return MISSING_PICK
    return value.isoformat(timespec="microseconds") + "Z"


def format_float(value, missing=""):
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return f"{number:.8g}"


def event_header(event, event_id):
    attrs = event.attrs
    return [
        format_time(parse_time(scalar_attr(attrs, "event_time"))),
        format_float(scalar_attr(attrs, "latitude")),
        format_float(scalar_attr(attrs, "longitude")),
        format_float(scalar_attr(attrs, "depth_km")),
        format_float(scalar_attr(attrs, "magnitude"), missing="-9"),
        str(event_id),
    ]


def matching_pick_indices(attrs, current_event_id):
    phase_count = len(list_attr(attrs, "phase_time"))
    pick_event_ids = list_attr(attrs, "event_id")
    if not pick_event_ids:
        return set(range(phase_count))
    current = str(current_event_id)
    if len(pick_event_ids) == 1 and phase_count > 1:
        return set(range(phase_count)) if str(pick_event_ids[0]) == current else set()
    return {
        index for index, pick_event_id in enumerate(pick_event_ids)
        if str(pick_event_id) == current
    }


def event_sort_key(h5, event_id):
    obj = h5[event_id]
    if not isinstance(obj, h5py.Group):
        return datetime.max
    event_time = parse_time(scalar_attr(obj.attrs, "event_time"))
    return event_time or datetime.max


def station_picks(station_dataset, current_event_id):
    attrs = station_dataset.attrs
    phase_types = list_attr(attrs, "phase_type")
    phase_times = list_attr(attrs, "phase_time")
    keep_indices = matching_pick_indices(attrs, current_event_id)

    picks = {"P": [], "S": []}
    for index, phase_type in enumerate(phase_types):
        if index not in keep_indices:
            continue
        phase = str(phase_type).strip().upper()[:1]
        if phase not in picks or index >= len(phase_times):
            continue
        phase_time = parse_time(phase_times[index])
        if phase_time is not None:
            picks[phase].append(phase_time)

    p_time = min(picks["P"]) if picks["P"] else None
    s_time = min(picks["S"]) if picks["S"] else None
    if p_time is None and s_time is None:
        return None

    distance_km = scalar_attr(attrs, "distance_km", "")
    return [
        station_dataset.name.rsplit("/", 1)[-1],
        format_time(p_time),
        format_time(s_time),
        format_float(distance_km),
    ]


def iter_h5_files(args):
    roots = []
    if args.nc_dir:
        roots.append(Path(args.nc_dir))
    if args.sc_dir:
        roots.append(Path(args.sc_dir))
    if not roots:
        ceed_root = Path(args.ceed_root)
        roots = [
            ceed_root / "quakeflow_nc" / "waveform_h5",
            ceed_root / "quakeflow_sc" / "waveform_h5",
        ]

    years = set(args.years or [])
    files = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing CEED waveform directory: {root}")
        for path in sorted(root.glob("*.h5")):
            if years and path.stem.split("_", 1)[0] not in years:
                continue
            files.append(path)
    return files


def extract_phase(files, phase_out, station_counts_out, progress_every):
    Path(phase_out).parent.mkdir(parents=True, exist_ok=True)
    if station_counts_out:
        Path(station_counts_out).parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    station_counts = {}

    with open(phase_out, "w", newline="") as phase_fp:
        writer = csv.writer(phase_fp, lineterminator="\n")
        for path in files:
            print(f"reading {path}", flush=True)
            with h5py.File(path, "r") as h5:
                for event_id in sorted(h5.keys(), key=lambda item: event_sort_key(h5, item)):
                    event = h5[event_id]
                    if not isinstance(event, h5py.Group):
                        continue
                    pick_rows = []
                    for station_key in sorted(event.keys()):
                        obj = event[station_key]
                        if not isinstance(obj, h5py.Dataset):
                            continue
                        row = station_picks(obj, event_id)
                        if row is None:
                            continue
                        pick_rows.append(row)

                    if not pick_rows:
                        counts["events_without_picks"] += 1
                        continue

                    writer.writerow(event_header(event, event_id))
                    for row in pick_rows:
                        writer.writerow(row)
                        stats = station_counts.setdefault(row[0], Counter())
                        stats["pick_rows"] += 1
                        if row[1] != MISSING_PICK:
                            counts["p_picks"] += 1
                            stats["p_rows"] += 1
                        if row[2] != MISSING_PICK:
                            counts["s_picks"] += 1
                            stats["s_rows"] += 1
                        if row[1] != MISSING_PICK and row[2] != MISSING_PICK:
                            stats["ps_rows"] += 1

                    counts["events_written"] += 1
                    counts["pick_rows_written"] += len(pick_rows)
                    if (
                        progress_every
                        and counts["events_written"] % progress_every == 0
                    ):
                        print(
                            "events written: "
                            f"{counts['events_written']:,}; "
                            f"pick rows: {counts['pick_rows_written']:,}",
                            flush=True,
                        )

    if station_counts_out:
        with open(station_counts_out, "w", newline="") as fp:
            writer = csv.writer(fp, lineterminator="\n")
            writer.writerow(["station_key", "pick_rows", "p_rows", "s_rows", "ps_rows"])
            rows = sorted(
                station_counts.items(),
                key=lambda item: (-item[1]["pick_rows"], item[0]),
            )
            for station_key, stats in rows:
                writer.writerow(
                    [
                        station_key,
                        stats["pick_rows"],
                        stats["p_rows"],
                        stats["s_rows"],
                        stats["ps_rows"],
                    ]
                )

    return counts


def main():
    from types import SimpleNamespace
    args = SimpleNamespace(
        ceed_root=str(CEED_ROOT), nc_dir=NC_DIR, sc_dir=SC_DIR, years=YEARS,
        phase_out=str(PHASE_OUT), station_counts_out=str(STATION_COUNTS_OUT),
        progress_every=PROGRESS_EVERY,
    )
    files = iter_h5_files(args)
    if not files:
        raise ValueError("No CEED HDF5 files matched the requested inputs.")

    print(f"HDF5 files: {len(files)}")
    counts = extract_phase(
        files,
        args.phase_out,
        args.station_counts_out,
        args.progress_every,
    )

    print(f"phase file: {args.phase_out}")
    if args.station_counts_out:
        print(f"station counts: {args.station_counts_out}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
