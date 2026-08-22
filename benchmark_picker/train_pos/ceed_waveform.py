"""Cut CEED HDF5 station waveforms into augmented 25 s SAC samples.

Input is the grouped training phase file from analyze_ceed_phase_feature_rarity.py:

    ot,lat,lon,dep,mag,event_id
    net.sta.loc.chn,tp,ts,epi_dist_km,hypo_dist_km,split,num_aug,...

For each station pick row, this script loads the corresponding 120 s CEED HDF5
station waveform, preprocesses it, randomly cuts num_aug 25 s windows containing
both P and S arrivals, normalizes each channel independently, and writes SAC.

Output layout:

    out_root/train/event_id_ot_yyyymmddhhmmss.ss/net.sta.chn.aug-idx.sac
    out_root/valid/event_id_ot_yyyymmddhhmmss.ss/net.sta.chn.aug-idx.sac

The phase file may not contain the original HDF5 group id if it was generated
before event ids were added to phase headers. In that case, events are resolved
by matching CEED event_time within a small tolerance.
"""

import bisect
import csv
import multiprocessing as mp
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import re

import h5py
import numpy as np
from obspy import Stream, Trace, UTCDateTime


MISSING_PICK = "-1"
DEFAULT_CEED_ROOT = Path("/nas/zhouyj/CEED")
DEFAULT_OUT_ROOT = Path("/nas/zhouyj/CEED_train_sac")


def parse_time(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text)


def utc(value):
    return UTCDateTime(value.isoformat())


def parse_pick_time(value):
    text = str(value).strip()
    if not text or text == MISSING_PICK:
        return None
    return parse_time(text)


def format_ot_for_dir(value):
    return value.strftime("%Y%m%d%H%M%S.%f")[:17]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))


def event_dir_name(event_id, ot):
    return f"{safe_name(event_id)}_ot_{format_ot_for_dir(ot)}"


def is_event_header(fields):
    if len(fields) < 5 or "T" not in fields[0]:
        return False
    try:
        parse_time(fields[0])
        float(fields[1]); float(fields[2]); float(fields[3]); float(fields[4])
    except ValueError:
        return False
    return True


def event_id_from_time(value):
    return value.strftime("%Y%m%d%H%M%S.%f")[:-3]


def datetime_seconds(value):
    day_seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    return value.toordinal() * 86400.0 + day_seconds


def read_training_phase(path):
    samples = []
    current = None
    counts = Counter()
    with open(path, newline="") as fp:
        reader = csv.reader(fp)
        for line_number, fields in enumerate(reader, 1):
            if not fields:
                continue
            fields = [item.strip() for item in fields]
            if is_event_header(fields):
                ot = parse_time(fields[0])
                current = {
                    "ot": ot,
                    "lat": float(fields[1]),
                    "lon": float(fields[2]),
                    "dep": float(fields[3]),
                    "mag": float(fields[4]),
                    "event_id": fields[5] if len(fields) > 5 and fields[5] else event_id_from_time(ot),
                }
                counts["events"] += 1
                continue

            if current is None or len(fields) < 7:
                counts["bad_pick_rows"] += 1
                continue

            tp = parse_pick_time(fields[1])
            ts = parse_pick_time(fields[2])
            if tp is None or ts is None:
                counts["skipped_missing_p_or_s"] += 1
                continue
            try:
                epi_dist_km = float(fields[3])
            except ValueError:
                epi_dist_km = np.nan
            try:
                hypo_dist_km = float(fields[4])
            except ValueError:
                hypo_dist_km = np.nan
            split = fields[5].lower()
            try:
                num_aug = int(float(fields[6]))
            except ValueError:
                num_aug = 1
            if split not in {"train", "valid"}:
                counts["skipped_bad_split"] += 1
                continue
            if num_aug <= 0:
                counts["skipped_nonpositive_num_aug"] += 1
                continue

            sample = dict(current)
            sample.update(
                {
                    "line_number": line_number,
                    "station_key": fields[0],
                    "tp": tp,
                    "ts": ts,
                    "epi_dist_km": epi_dist_km,
                    "hypo_dist_km": hypo_dist_km,
                    "split": split,
                    "num_aug": num_aug,
                }
            )
            samples.append(sample)
            counts["samples"] += 1
    if not samples:
        raise ValueError(f"No usable samples read from {path}")
    return samples, counts


def ceed_h5_roots(args):
    roots = []
    if args.nc_dir:
        roots.append(Path(args.nc_dir))
    if args.sc_dir:
        roots.append(Path(args.sc_dir))
    if roots:
        return roots
    root = Path(args.ceed_root)
    return [root / "quakeflow_nc" / "waveform_h5", root / "quakeflow_sc" / "waveform_h5"]


def year_h5_files(roots, year):
    files = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing CEED waveform directory: {root}")
        files.extend(sorted(root.glob(f"{year}*.h5")))
    return files


def attr_scalar(attrs, name, default=None):
    if name not in attrs:
        return default
    value = attrs[name]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_event_id_index(files, needed_ids):
    """Map requested CEED HDF5 event ids to files without reading attrs."""
    needed_ids = set(str(item) for item in needed_ids)
    by_id = {}
    for path in files:
        with h5py.File(path, "r") as h5:
            for event_id in h5.keys():
                event_id = str(event_id)
                if event_id in needed_ids and event_id not in by_id:
                    by_id[event_id] = (path, event_id, None)
        if len(by_id) == len(needed_ids):
            break
    return by_id


def build_year_index(files, needed_times=None):
    """Build slow event_time index, used only as fallback for missing event ids."""
    time_rows = []
    needed_min = None
    needed_max = None
    if needed_times:
        values = [datetime_seconds(item) for item in needed_times]
        needed_min = min(values)
        needed_max = max(values)
    for path in files:
        with h5py.File(path, "r") as h5:
            for event_id in h5.keys():
                obj = h5[event_id]
                if not isinstance(obj, h5py.Group):
                    continue
                event_time = attr_scalar(obj.attrs, "event_time")
                if not event_time:
                    continue
                try:
                    ot = parse_time(event_time)
                except ValueError:
                    continue
                ot_sec = datetime_seconds(ot)
                if needed_min is not None and (ot_sec < needed_min - 60.0 or ot_sec > needed_max + 60.0):
                    continue
                item = (path, event_id, ot)
                time_rows.append((ot_sec, item))
    time_rows.sort(key=lambda item: item[0])
    time_values = [item[0] for item in time_rows]
    return time_values, time_rows


def resolve_samples(samples, roots, time_tolerance_sec):
    resolved = []
    counts = Counter()
    by_year = defaultdict(list)
    for sample in samples:
        by_year[sample["ot"].year].append(sample)

    for year in sorted(by_year):
        year_samples = by_year[year]
        files = year_h5_files(roots, year)
        if not files:
            counts["missing_year_files"] += len(year_samples)
            print(f"resolve {year}: no HDF5 files for {len(year_samples):,} rows", flush=True)
            continue

        needed_ids = {sample["event_id"] for sample in year_samples}
        by_id = build_event_id_index(files, needed_ids)
        missing = [sample for sample in year_samples if sample["event_id"] not in by_id]

        time_values, time_rows = [], []
        if missing:
            time_values, time_rows = build_year_index(files, [sample["ot"] for sample in missing])

        for sample in year_samples:
            match = by_id.get(sample["event_id"])
            if match is None and time_rows:
                target = datetime_seconds(sample["ot"])
                lo = bisect.bisect_left(time_values, target - time_tolerance_sec)
                hi = bisect.bisect_right(time_values, target + time_tolerance_sec)
                if hi > lo:
                    candidates = [
                        (abs(time_rows[i][0] - target), time_rows[i][1])
                        for i in range(lo, hi)
                    ]
                    candidates.sort(key=lambda item: item[0])
                    match = candidates[0][1]
                    counts["resolved_by_time"] += 1
            elif match is not None:
                counts["resolved_by_event_id"] += 1

            if match is None:
                counts["unresolved_event"] += 1
                continue
            sample["h5_path"] = match[0]
            sample["h5_event_id"] = match[1]
            resolved.append(sample)
            counts["resolved_samples"] += 1
        print(
            f"resolve {year}: rows={len(year_samples):,}, by_id={len(year_samples) - len(missing):,}, "
            f"fallback={len(missing):,}, total={len(resolved):,}",
            flush=True,
        )
    if not resolved:
        raise ValueError("No phase samples could be resolved to CEED HDF5 events")
    return resolved, counts


def split_station_key(station_key):
    parts = station_key.split(".")
    net = parts[0] if len(parts) > 0 else ""
    sta = parts[1] if len(parts) > 1 else ""
    loc = parts[2] if len(parts) > 2 else ""
    band = parts[3] if len(parts) > 3 else (parts[-1] if parts else "")
    return net, sta, loc, band


def is_acceleration_channel(channel_or_band):
    code = str(channel_or_band).upper()
    return len(code) >= 2 and code[1] == "N"


def component_channels(band, component_attr):
    components = str(component_attr or "ENZ")
    if len(components) < 3:
        components = "ENZ"
    return [f"{band}{comp}" for comp in components[:3]]


def make_stream(dataset, sample, args):
    data = np.asarray(dataset[:, :], dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != 3:
        raise ValueError(f"Expected waveform shape (3, nt), got {data.shape}")

    attrs = dataset.attrs
    event_attrs = dataset.parent.attrs
    net, sta, loc, band = split_station_key(sample["station_key"])
    channels = component_channels(band, attr_scalar(attrs, "component", "ENZ"))
    sampling_rate = float(attr_scalar(event_attrs, "sampling_rate", attr_scalar(attrs, "sampling_rate", 100.0)))
    begin_time = parse_time(attr_scalar(event_attrs, "begin_time", attr_scalar(attrs, "begin_time")))
    start = utc(begin_time)

    stream = Stream()
    for i, channel in enumerate(channels):
        tr = Trace(data=np.asarray(data[i], dtype=np.float64))
        tr.stats.network = net
        tr.stats.station = sta
        tr.stats.location = loc
        tr.stats.channel = channel
        tr.stats.starttime = start
        tr.stats.sampling_rate = sampling_rate
        tr.stats.sac = {}
        if np.isfinite(sample.get("lat", np.nan)):
            tr.stats.sac.evla = float(sample["lat"])
        if np.isfinite(sample.get("lon", np.nan)):
            tr.stats.sac.evlo = float(sample["lon"])
        if np.isfinite(sample.get("dep", np.nan)):
            tr.stats.sac.evdp = float(sample["dep"])
        if np.isfinite(sample.get("mag", np.nan)):
            tr.stats.sac.mag = float(sample["mag"])
        if np.isfinite(sample.get("epi_dist_km", np.nan)):
            tr.stats.sac.dist = float(sample["epi_dist_km"])
        sta_lat = attr_scalar(attrs, "latitude")
        sta_lon = attr_scalar(attrs, "longitude")
        sta_ele = attr_scalar(attrs, "elevation_m")
        if sta_lat is not None:
            tr.stats.sac.stla = float(sta_lat)
        if sta_lon is not None:
            tr.stats.sac.stlo = float(sta_lon)
        if sta_ele is not None:
            tr.stats.sac.stel = float(sta_ele)
        stream.append(tr)

    if args.integrate_acceleration and is_acceleration_channel(band):
        sample["integrated_acceleration"] = True
        for tr in stream:
            tr.integrate(method="cumtrapz")
    else:
        sample["integrated_acceleration"] = False

    return stream


def preprocess_stream(stream, args):
    st = stream.copy()
    for tr in st:
        tr.data = np.asarray(tr.data, dtype=np.float64)
        tr.data[np.isnan(tr.data)] = 0.0
        tr.data[np.isinf(tr.data)] = 0.0
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=args.taper_max_percentage, max_length=args.taper_max_length)
    st.filter("bandpass", freqmin=args.freqmin, freqmax=args.freqmax, corners=args.filter_corners, zerophase=True)
    return st


def valid_window_start_range(stream, tp, ts, args):
    record_start = max(tr.stats.starttime for tr in stream)
    record_end = min(tr.stats.endtime for tr in stream)
    tp_utc = utc(tp)
    ts_utc = utc(ts)
    if tp_utc > ts_utc:
        return None
    min_start = max(record_start, ts_utc - args.window_length + args.phase_margin)
    max_start = min(tp_utc - args.phase_margin, record_end - args.window_length)
    if min_start > max_start:
        return None
    return min_start, max_start


def cut_and_normalize(stream, start_time, args):
    end_time = start_time + args.window_length
    st = stream.copy().slice(start_time, end_time, nearest_sample=True)
    npts = int(round(args.window_length * args.sample_rate))
    for tr in st:
        if abs(tr.stats.sampling_rate - args.sample_rate) > 1e-6:
            tr.resample(args.sample_rate)
        if len(tr.data) < npts:
            padded = np.zeros(npts, dtype=np.float32)
            padded[:len(tr.data)] = tr.data.astype(np.float32)
            tr.data = padded
        elif len(tr.data) > npts:
            tr.data = tr.data[:npts]
        tr.stats.starttime = start_time
        tr.stats.sampling_rate = args.sample_rate
        max_abs = float(np.max(np.abs(tr.data))) if len(tr.data) else 0.0
        if max_abs > 0:
            tr.data = (tr.data / max_abs).astype(np.float32)
        else:
            tr.data = tr.data.astype(np.float32)
    return st


def update_sac_time_headers(trace, p_rel, s_rel):
    t0 = trace.stats.starttime
    if "sac" not in trace.stats:
        trace.stats.sac = {}
    trace.stats.sac.nzyear = t0.year
    trace.stats.sac.nzjday = t0.julday
    trace.stats.sac.nzhour = t0.hour
    trace.stats.sac.nzmin = t0.minute
    trace.stats.sac.nzsec = t0.second
    trace.stats.sac.nzmsec = int(t0.microsecond / 1000)
    trace.stats.sac.t0 = float(p_rel)
    trace.stats.sac.t1 = float(s_rel)


def write_sac_sample(stream, sample, aug_idx, start_time, args):
    out_dir = Path(args.out_root) / sample["split"] / event_dir_name(sample["event_id"], sample["ot"])
    out_dir.mkdir(parents=True, exist_ok=True)
    p_rel = utc(sample["tp"]) - start_time
    s_rel = utc(sample["ts"]) - start_time
    paths = []
    for tr in stream:
        update_sac_time_headers(tr, p_rel, s_rel)
        filename = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}.aug-{aug_idx}.sac"
        out_path = out_dir / filename
        tr.write(str(out_path), format="SAC")
        paths.append(str(out_path))
    return paths


def process_sample(dataset, sample, args, rng):
    counts = Counter()
    written_paths = []
    try:
        stream = make_stream(dataset, sample, args)
        if sample.get("integrated_acceleration"):
            counts["acceleration_records_integrated"] += 1
        if abs(stream[0].stats.sampling_rate - args.sample_rate) > 1e-6:
            stream.resample(args.sample_rate)
        stream = preprocess_stream(stream, args)
        window_range = valid_window_start_range(stream, sample["tp"], sample["ts"], args)
        if window_range is None:
            counts["skipped_no_valid_window"] += 1
            return counts, written_paths
        min_start, max_start = window_range
        span = max_start - min_start
        for aug_idx in range(sample["num_aug"]):
            offset = rng.random() * span if span > 0 else 0.0
            start_time = min_start + offset
            cut = cut_and_normalize(stream, start_time, args)
            if len(cut) != 3:
                counts["skipped_bad_cut_channel_count"] += 1
                continue
            paths = write_sac_sample(cut, sample, aug_idx, start_time, args)
            written_paths.append(paths)
            counts["sac_files_written"] += len(paths)
            counts["augmented_samples_written"] += 1
    except Exception as exc:
        counts["skipped_exception"] += 1
        if args.verbose_errors:
            print(f"sample line {sample.get('line_number')} failed: {exc}", flush=True)
    return counts, written_paths


def process_h5_group(task):
    group_index, h5_path, samples, args = task
    counts = Counter()
    train_paths = []
    valid_paths = []
    rng = np.random.default_rng(args.random_seed + group_index)
    print(f"reading {h5_path}", flush=True)
    with h5py.File(h5_path, "r") as h5:
        by_event = defaultdict(list)
        for sample in samples:
            by_event[sample["h5_event_id"]].append(sample)
        done = 0
        for event_id in sorted(by_event):
            if event_id not in h5:
                counts["missing_event_in_h5"] += len(by_event[event_id])
                continue
            event = h5[event_id]
            for sample in by_event[event_id]:
                station_key = sample["station_key"]
                if station_key not in event:
                    counts["missing_station_dataset"] += 1
                    continue
                sample_counts, paths = process_sample(event[station_key], sample, args, rng)
                counts.update(sample_counts)
                if paths:
                    if sample["split"] == "train":
                        train_paths.extend(paths)
                    else:
                        valid_paths.extend(paths)
                done += 1
                if args.progress_every and done % args.progress_every == 0:
                    print(f"{h5_path}: processed {done:,}/{len(samples):,}", flush=True)
    counts["phase_samples_processed"] += len(samples)
    return counts, train_paths, valid_paths


def process_samples(samples, args):
    counts = Counter()
    train_paths = []
    valid_paths = []
    by_path = defaultdict(list)
    for sample in samples:
        by_path[str(Path(sample["h5_path"]))].append(sample)
    tasks = [(idx, h5_path, by_path[h5_path], args) for idx, h5_path in enumerate(sorted(by_path))]

    if args.num_workers <= 1 or len(tasks) == 1:
        results = [process_h5_group(task) for task in tasks]
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(pool.imap_unordered(process_h5_group, tasks))

    for group_counts, group_train_paths, group_valid_paths in results:
        counts.update(group_counts)
        train_paths.extend(group_train_paths)
        valid_paths.extend(group_valid_paths)
    return counts, train_paths, valid_paths


def path_array(paths):
    if not paths:
        return np.empty((0, 3), dtype="U1")
    return np.asarray(paths, dtype=str)


def save_path_lists(out_root, train_paths, valid_paths):
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "train_pos.npy", path_array(train_paths))
    np.save(root / "valid_pos.npy", path_array(valid_paths))


def write_summary(path, phase_counts, resolve_counts, process_counts, args, n_samples, n_resolved):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["parameter", "value"])
        writer.writerow(["phase_file", args.phase_file])
        writer.writerow(["ceed_root", args.ceed_root])
        writer.writerow(["out_root", args.out_root])
        writer.writerow(["window_length", args.window_length])
        writer.writerow(["sample_rate", args.sample_rate])
        writer.writerow(["freqmin", args.freqmin])
        writer.writerow(["freqmax", args.freqmax])
        writer.writerow(["phase_margin", args.phase_margin])
        writer.writerow(["random_seed", args.random_seed])
        writer.writerow(["num_workers", args.num_workers])
        writer.writerow(["samples_read", n_samples])
        writer.writerow(["samples_resolved", n_resolved])
        for prefix, counts in [("phase", phase_counts), ("resolve", resolve_counts), ("process", process_counts)]:
            for key in sorted(counts):
                writer.writerow([f"{prefix}_{key}", counts[key]])

