"""Analyze PAL station-pick rarity and build num_aug-aware phase files.

Input is a PAL phase file plus the matching PAL station file:

    ot,lat,lon,dep,mag
    net.sta[.chn],tp,ts,...

Epicentral distance is calculated from event and station coordinates. This
script computes the third rarity feature as hypocentral distance:

    hypo_dist_km = sqrt(epi_dist_km ** 2 + (dep_km + elev_km) ** 2)

Each station pick row is treated as one single-station training sample. The
three rarity features are FMD/magnitude, spatiotemporal seismicity rate, and
hypocentral distance. Dataset train/validation assignment remains the
responsibility of the downstream sample cutter; this analysis only determines
the training augmentation count for each pick.

Main outputs:
  * feature CSV with one row per station waveform
  * figure of feature distributions
  * figure of joint probability, rarity, and num_aug divisions
  * CSV training phase table, one station waveform per row
  * PAL-like grouped training phase file with tagged per-pick num_aug metadata
"""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
except ImportError:
    gaussian_filter = None
    gaussian_filter1d = None


MISSING_PICK = "-1"
RARITY_FLOOR = 1e-12
PROB_HIST_BINS = 80

# Internal numerical/plotting defaults. These are deliberately kept out of the
# case config: they stabilize density estimates but do not define the study's
# physical resolution or augmentation policy.
MAG_SMOOTH_SIGMA_BINS = 1.0
HYPO_DIST_SMOOTH_SIGMA_BINS = 1.5
SPATIOTEMPORAL_COUNT_LOG_BIN_WIDTH = 0.1
SPATIOTEMPORAL_COUNT_SMOOTH_SIGMA_BINS = 1.5
SPATIOTEMPORAL_DENSITY_SMOOTH_SIGMA_XY_BINS = 1.5
SPATIOTEMPORAL_DENSITY_SMOOTH_SIGMA_TIME_BINS = 1.0
SPATIOTEMPORAL_DENSITY_FLOOR = 1e-30
SPATIOTEMPORAL_SCALED_FLOOR = 1e-6
FIGURE_DPI = 300
PHASE_REQUIREMENT = "any"
STATION_SPLIT_LEVEL = "net_sta"
VALIDATION_FRACTION = 0.0
RANDOM_SEED = 20250630
VALIDATION_NUM_AUG = 1


def parse_time(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y%m%d%H%M%S.%f", "%Y%m%d%H%M%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        raise


def format_time(value):
    return value.isoformat(timespec="microseconds") + "Z"


def datetime_days(value):
    return (
        value.toordinal()
        + (value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6) / 86400.0
    )


def event_id_from_time(value):
    return value.strftime("%Y%m%d%H%M%S.%f")[:-3]


def is_event_header(fields):
    if len(fields) < 5:
        return False
    try:
        parse_time(fields[0])
        float(fields[1]); float(fields[2]); float(fields[3]); float(fields[4])
    except ValueError:
        return False
    return True


def parse_pick_time(value):
    text = str(value).strip()
    if not text or text == MISSING_PICK:
        return None
    return parse_time(text)


def station_split_key(station_key, level):
    parts = station_key.split(".")
    if level == "net_sta":
        return ".".join(parts[:2]) if len(parts) >= 2 else station_key
    if level == "net_sta_loc_chn":
        return station_key
    raise ValueError(f"Unsupported station split level: {level}")


def project(lon, lat, lon0, lat0):
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (np.asarray(lat) - lat0) * 111.32
    return x, y


def smooth1d(counts, sigma):
    counts = np.asarray(counts, dtype=float)
    if gaussian_filter1d is not None and sigma > 0:
        return gaussian_filter1d(counts, sigma=sigma, mode="nearest")
    if sigma <= 0:
        return counts
    half_width = max(1, int(np.ceil(4 * sigma)))
    offsets = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    # numpy's "same" returns max(len(signal), len(kernel)); keep the histogram
    # length when a broad smoothing kernel is used on a small dataset.
    padded = np.pad(counts, (half_width, half_width), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_edges(values, width, min_value=None):
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values)) if min_value is None else min_value
    hi = float(np.nanmax(values))
    lo = math.floor(lo / width) * width
    hi = math.ceil(hi / width) * width
    if hi <= lo:
        hi = lo + width
    return np.arange(lo, hi + 1.5 * width, width)


def hist_pdf(values, edges, sigma):
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    smooth = smooth1d(counts, sigma)
    area = float(np.sum(smooth * widths))
    if area > 0:
        pdf = smooth / area
    else:
        pdf = np.ones_like(smooth, dtype=float) / np.sum(widths)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, len(pdf) - 1)
    return np.maximum(pdf[idx], RARITY_FLOOR), pdf, counts


def log_scale_positive(values, floor, scaled_floor):
    values = np.asarray(values, dtype=float)
    logs = np.log10(np.maximum(values, floor))
    finite = np.isfinite(logs)
    scaled = np.ones_like(logs, dtype=float)
    if np.any(finite):
        lo = float(np.min(logs[finite]))
        hi = float(np.max(logs[finite]))
        if hi > lo:
            scaled[finite] = (logs[finite] - lo) / (hi - lo)
            scaled[finite] = scaled_floor + (1.0 - scaled_floor) * scaled[finite]
        else:
            scaled[finite] = 1.0
        scaled[~finite] = scaled_floor
    return logs, np.maximum(scaled, scaled_floor)


def read_station_coordinates(path):
    stations = {}
    with open(path, newline="") as fp:
        for line_number, fields in enumerate(csv.reader(fp), 1):
            if not fields or not fields[0].strip():
                continue
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number} invalid station row")
            key = fields[0].strip()
            coordinates = tuple(float(value) for value in fields[1:4])
            stations[key] = coordinates
            stations.setdefault(".".join(key.split(".")[:2]), coordinates)
    if not stations:
        raise ValueError(f"No stations read from {path}")
    return stations


def station_coordinates(stations, station_key):
    parts = station_key.split(".")
    for candidate in (station_key, ".".join(parts[:2])):
        if candidate in stations:
            return stations[candidate]
    return None


def epicentral_distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    angle = 2.0 * np.arctan2(
        np.sqrt(value), np.sqrt(max(0.0, 1.0 - value))
    )
    return float(6371.0 * angle)


def read_pal_phase(path, phase_requirement, stations):
    samples = []
    event_headers = []
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
                    "event_index": len(event_headers),
                    "event_id": fields[5] if len(fields) > 5 and fields[5] else event_id_from_time(ot),
                    "line_number": line_number,
                    "ot": ot,
                    "lat": float(fields[1]),
                    "lon": float(fields[2]),
                    "dep": float(fields[3]),
                    "mag": float(fields[4]),
                    "fields": fields[:5],
                }
                event_headers.append(current)
                counts["events"] += 1
                continue

            if current is None or len(fields) < 3:
                counts["bad_pick_rows"] += 1
                continue

            tp = parse_pick_time(fields[1])
            ts = parse_pick_time(fields[2])
            has_p = tp is not None
            has_s = ts is not None
            if phase_requirement == "both" and not (has_p and has_s):
                counts["skipped_missing_required_phase"] += 1
                continue
            if phase_requirement == "p" and not has_p:
                counts["skipped_missing_required_phase"] += 1
                continue
            if phase_requirement == "s" and not has_s:
                counts["skipped_missing_required_phase"] += 1
                continue
            if phase_requirement == "any" and not (has_p or has_s):
                counts["skipped_missing_required_phase"] += 1
                continue

            station_key = fields[0]
            coordinates = station_coordinates(stations, station_key)
            if coordinates is None:
                counts["skipped_missing_station"] += 1
                continue
            sta_lat, sta_lon, sta_elev_m = coordinates
            epi_dist_km = epicentral_distance_km(
                current["lat"], current["lon"], sta_lat, sta_lon
            )
            parts = station_key.split(".")
            samples.append(
                {
                    "sample_index": len(samples),
                    "event_index": current["event_index"],
                    "event_id": current["event_id"],
                    "event_line_number": current["line_number"],
                    "ot": current["ot"],
                    "lat": current["lat"],
                    "lon": current["lon"],
                    "dep": current["dep"],
                    "mag": current["mag"],
                    "station_key": station_key,
                    "net_sta": ".".join(parts[:2]) if len(parts) >= 2 else station_key,
                    "tp": tp,
                    "ts": ts,
                    "has_p": has_p,
                    "has_s": has_s,
                    "epi_dist_km": epi_dist_km,
                    "dist_km": epi_dist_km,
                    "station_elev_km": sta_elev_m / 1000.0,
                }
            )
            counts["samples"] += 1

    if not samples:
        raise ValueError(f"No usable station-waveform samples read from {path}")
    return event_headers, samples, counts


def spatiotemporal_density_for_samples(samples, spatial_bin_km, time_bin_days, smooth_xy, smooth_t):
    events = []
    seen = set()
    for sample in sorted(samples, key=lambda item: item["event_index"]):
        event_index = sample["event_index"]
        if event_index in seen:
            continue
        seen.add(event_index)
        events.append(sample)

    lats = np.array([s["lat"] for s in events], dtype=float)
    lons = np.array([s["lon"] for s in events], dtype=float)
    time_days_abs = np.array([datetime_days(s["ot"]) for s in events], dtype=float)
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    event_x, event_y = project(lons, lats, lon0, lat0)
    event_time_days = time_days_abs - float(np.min(time_days_abs))

    xe = np.arange(np.min(event_x) - spatial_bin_km, np.max(event_x) + 2 * spatial_bin_km, spatial_bin_km)
    ye = np.arange(np.min(event_y) - spatial_bin_km, np.max(event_y) + 2 * spatial_bin_km, spatial_bin_km)
    te = np.arange(0.0, np.max(event_time_days) + 1.5 * time_bin_days, time_bin_days)
    if len(te) < 2:
        te = np.array([0.0, time_bin_days])

    ix = np.clip(np.searchsorted(xe, event_x, side="right") - 1, 0, len(xe) - 2)
    iy = np.clip(np.searchsorted(ye, event_y, side="right") - 1, 0, len(ye) - 2)
    it = np.clip(np.searchsorted(te, event_time_days, side="right") - 1, 0, len(te) - 2)
    shape = (len(xe) - 1, len(ye) - 1, len(te) - 1)
    linear = np.ravel_multi_index((ix, iy, it), shape)
    occupied, inverse, counts = np.unique(linear, return_inverse=True, return_counts=True)
    event_bin_counts = counts[inverse].astype(float)

    grid = np.zeros(shape, dtype=np.float32)
    np.add.at(grid.ravel(), occupied, counts.astype(np.float32))
    if gaussian_filter is not None and (smooth_xy > 0 or smooth_t > 0):
        smooth = gaussian_filter(grid, sigma=(smooth_xy, smooth_xy, smooth_t), mode="constant", cval=0.0)
    else:
        smooth = grid
    cell_volume = spatial_bin_km * spatial_bin_km * time_bin_days
    total = float(np.sum(smooth) * cell_volume)
    if total > 0:
        density = smooth / total
    else:
        density = np.ones(shape, dtype=np.float32) / (np.prod(shape) * cell_volume)
    event_density = density[ix, iy, it].astype(float)

    by_event = {}
    for i, event in enumerate(events):
        by_event[event["event_index"]] = (
            float(event_x[i]),
            float(event_y[i]),
            float(event_time_days[i]),
            float(event_bin_counts[i]),
            float(event_density[i]),
        )

    sample_values = [by_event[sample["event_index"]] for sample in samples]
    x, y, time_days, bin_counts, sample_density = [np.array(col, dtype=float) for col in zip(*sample_values)]
    return x, y, time_days, bin_counts, sample_density


def filter_samples_by_hypo_distance(samples, max_hypo_dist_km, counts):
    if max_hypo_dist_km is None or max_hypo_dist_km <= 0:
        return samples

    epi_dist = np.array([s["epi_dist_km"] for s in samples], dtype=float)
    dep = np.array([
        s["dep"] + s.get("station_elev_km", 0.0) for s in samples
    ], dtype=float)
    finite_dist = np.isfinite(epi_dist)
    if not np.all(finite_dist):
        fill = float(np.nanmedian(epi_dist[finite_dist])) if np.any(finite_dist) else 0.0
        epi_dist = np.where(finite_dist, epi_dist, fill)
    hypo_dist = np.sqrt(np.maximum(epi_dist, 0.0) ** 2 + np.maximum(dep, 0.0) ** 2)

    kept = []
    for index, sample in enumerate(samples):
        sample["dist_km"] = float(epi_dist[index])
        sample["epi_dist_km"] = float(epi_dist[index])
        sample["hypo_dist_km"] = float(hypo_dist[index])
        if hypo_dist[index] <= max_hypo_dist_km:
            sample["sample_index"] = len(kept)
            kept.append(sample)

    counts["samples_before_hypo_distance_filter"] = len(samples)
    counts["samples_skipped_hypo_dist_gt_max"] = len(samples) - len(kept)
    counts["samples_after_hypo_distance_filter"] = len(kept)
    counts["max_hypo_dist_km"] = max_hypo_dist_km
    if not kept:
        raise ValueError(f"No samples remain after hypo_dist_km <= {max_hypo_dist_km:g} km filter")
    return kept


def assign_station_splits(samples, validation_fraction, random_seed, split_level):
    keys = sorted({station_split_key(s["station_key"], split_level) for s in samples})
    rng = np.random.default_rng(random_seed)
    shuffled = np.array(keys, dtype=object)
    rng.shuffle(shuffled)
    n_valid = int(round(len(shuffled) * validation_fraction))
    if validation_fraction > 0 and len(shuffled) > 1:
        n_valid = max(1, min(len(shuffled) - 1, n_valid))
    valid_keys = set(shuffled[:n_valid])
    split_by_key = {key: ("valid" if key in valid_keys else "train") for key in keys}
    for sample in samples:
        sample["split_key"] = station_split_key(sample["station_key"], split_level)
        sample["split"] = split_by_key[sample["split_key"]]
    return split_by_key


def compute_rarity(samples, args):
    mag = np.array([s["mag"] for s in samples], dtype=float)
    epi_dist = np.array([s["epi_dist_km"] for s in samples], dtype=float)
    dep = np.array([
        s["dep"] + s.get("station_elev_km", 0.0) for s in samples
    ], dtype=float)

    finite_dist = np.isfinite(epi_dist)
    if not np.all(finite_dist):
        fill = float(np.nanmedian(epi_dist[finite_dist])) if np.any(finite_dist) else 0.0
        epi_dist = np.where(finite_dist, epi_dist, fill)
    hypo_dist = np.sqrt(np.maximum(epi_dist, 0.0) ** 2 + np.maximum(dep, 0.0) ** 2)

    x, y, time_days, st_count, st_density = spatiotemporal_density_for_samples(
        samples,
        args.spatial_bin_km,
        args.spatiotemporal_time_bin_days,
        args.spatiotemporal_density_smooth_sigma_xy_bins,
        args.spatiotemporal_density_smooth_sigma_time_bins,
    )
    st_feature_count = np.log10(np.maximum(st_count, 1.0))
    st_count_edges = make_edges(st_feature_count, args.spatiotemporal_count_log_bin_width, min_value=0.0)

    mag_edges = make_edges(mag, args.mag_bin_width)
    hypo_edges = make_edges(hypo_dist, args.hypo_dist_bin_width, min_value=0.0)

    p_mag, mag_pdf, mag_counts = hist_pdf(mag, mag_edges, args.mag_hist_smooth_sigma_bins)
    p_hypo, hypo_pdf, hypo_counts = hist_pdf(hypo_dist, hypo_edges, args.hypo_dist_hist_smooth_sigma_bins)
    _p_st_count, st_count_pdf, st_count_counts = hist_pdf(
        st_feature_count,
        st_count_edges,
        args.spatiotemporal_count_hist_smooth_sigma_bins,
    )
    st_log10_density, p_st = log_scale_positive(
        st_density,
        args.spatiotemporal_density_floor,
        args.spatiotemporal_scaled_floor,
    )

    p_joint = p_mag * p_hypo * p_st
    max_p = float(np.max(p_joint)) if len(p_joint) else 1.0
    p_norm = p_joint / max_p if max_p > 0 else np.ones_like(p_joint)
    rarity = -np.log10(np.maximum(p_norm, RARITY_FLOOR))
    thresholds = np.percentile(rarity, args.rarity_percentiles)

    num_aug_base = np.ones(len(samples), dtype=np.int32) * int(args.augmentation_values[0])
    for threshold, value in zip(thresholds, args.augmentation_values[1:]):
        num_aug_base[rarity >= threshold] = int(value)

    for i, sample in enumerate(samples):
        sample.update(
            {
                "dist_km": float(epi_dist[i]),
                "epi_dist_km": float(epi_dist[i]),
                "hypo_dist_km": float(hypo_dist[i]),
                "x_km": float(x[i]),
                "y_km": float(y[i]),
                "time_days_from_start": float(time_days[i]),
                "spatiotemporal_count": float(st_count[i]),
                "spatiotemporal_log10_count": float(st_feature_count[i]),
                "spatiotemporal_density": float(st_density[i]),
                "spatiotemporal_log10_density": float(st_log10_density[i]),
                "p_mag": float(p_mag[i]),
                "p_hypo_dist": float(p_hypo[i]),
                "p_spatiotemporal": float(p_st[i]),
                "p_joint": float(p_joint[i]),
                "p_norm": float(p_norm[i]),
                "rarity": float(rarity[i]),
                "num_aug_base": int(num_aug_base[i]),
            }
        )

    hist = {
        "mag": (mag, mag_edges, mag_pdf, mag_counts),
        "hypo_dist": (hypo_dist, hypo_edges, hypo_pdf, hypo_counts),
        "st_count": (st_feature_count, st_count_edges, st_count_pdf, st_count_counts),
        "st_density": st_density,
        "p_norm": p_norm,
        "rarity": rarity,
        "thresholds": thresholds,
    }
    return thresholds, hist


def apply_num_aug(samples, validation_num_aug):
    for sample in samples:
        if sample["split"] == "valid":
            sample["num_aug"] = int(validation_num_aug)
        else:
            sample["num_aug"] = int(sample["num_aug_base"])


def write_feature_csv(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_index", "event_index", "event_id", "event_line_number", "ot", "lat", "lon", "dep", "mag",
        "station_key", "net_sta", "split_key", "split", "tp", "ts", "has_p", "has_s",
        "dist_km", "epi_dist_km", "hypo_dist_km", "x_km", "y_km", "time_days_from_start",
        "spatiotemporal_count", "spatiotemporal_log10_count", "spatiotemporal_density",
        "spatiotemporal_log10_density", "p_mag", "p_hypo_dist", "p_spatiotemporal", "p_joint", "p_norm",
        "rarity", "num_aug_base", "num_aug",
    ]
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            row = {key: sample.get(key, "") for key in columns}
            row["ot"] = format_time(sample["ot"])
            row["tp"] = format_time(sample["tp"]) if sample["tp"] is not None else MISSING_PICK
            row["ts"] = format_time(sample["ts"]) if sample["ts"] is not None else MISSING_PICK
            writer.writerow(row)


def write_training_csv(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "event_id", "ot", "lat", "lon", "dep", "mag", "station_key", "net_sta", "tp", "ts",
        "dist_km", "epi_dist_km", "hypo_dist_km", "split", "is_train", "num_aug", "num_aug_base",
        "rarity", "p_norm", "has_p", "has_s",
    ]
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "event_id": sample["event_id"],
                    "ot": format_time(sample["ot"]),
                    "lat": sample["lat"],
                    "lon": sample["lon"],
                    "dep": sample["dep"],
                    "mag": sample["mag"],
                    "station_key": sample["station_key"],
                    "net_sta": sample["net_sta"],
                    "tp": format_time(sample["tp"]) if sample["tp"] is not None else MISSING_PICK,
                    "ts": format_time(sample["ts"]) if sample["ts"] is not None else MISSING_PICK,
                    "dist_km": sample["epi_dist_km"],
                    "epi_dist_km": sample["epi_dist_km"],
                    "hypo_dist_km": sample["hypo_dist_km"],
                    "split": sample["split"],
                    "is_train": 1 if sample["split"] == "train" else 0,
                    "num_aug": sample["num_aug"],
                    "num_aug_base": sample["num_aug_base"],
                    "rarity": f"{sample['rarity']:.6f}",
                    "p_norm": f"{sample['p_norm']:.8e}",
                    "has_p": int(sample["has_p"]),
                    "has_s": int(sample["has_s"]),
                }
            )


def write_training_pha(event_headers, samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    by_event = defaultdict(list)
    for sample in samples:
        by_event[sample["event_index"]].append(sample)

    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        for event in event_headers:
            event_samples = by_event.get(event["event_index"], [])
            if not event_samples:
                continue
            writer.writerow(event["fields"] + [event["event_id"]])
            for sample in sorted(event_samples, key=lambda item: item["station_key"]):
                writer.writerow(
                    [
                        sample["station_key"],
                        format_time(sample["tp"]) if sample["tp"] is not None else MISSING_PICK,
                        format_time(sample["ts"]) if sample["ts"] is not None else MISSING_PICK,
                        f"num_aug={sample['num_aug_base']}",
                        f"num_aug_base={sample['num_aug_base']}",
                        f"rarity={sample['rarity']:.6f}",
                        f"p_norm={sample['p_norm']:.8e}",
                        f"epi_dist_km={sample['epi_dist_km']:.4f}",
                        f"hypo_dist_km={sample['hypo_dist_km']:.4f}",
                    ]
                )


def write_summary(path, args, counts, split_by_key, thresholds, samples):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(s["split"] for s in samples)
    aug_counts = Counter(s["num_aug"] for s in samples)
    train_augmented = sum(s["num_aug"] for s in samples if s["split"] == "train")
    valid_augmented = sum(s["num_aug"] for s in samples if s["split"] == "valid")
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["parameter", "value"])
        writer.writerow(["phase_in", args.phase_in])
        writer.writerow(["feature_csv_out", args.feature_csv_out])
        writer.writerow(["training_csv_out", args.training_csv_out])
        writer.writerow(["training_pha_out", args.training_pha_out])
        writer.writerow(["feature_fig_out", args.feature_fig_out])
        writer.writerow(["prob_fig_out", args.prob_fig_out])
        writer.writerow(["station_file", args.station_file])
        writer.writerow(["phase_pick_distance_source", "event_and_station_coordinates"])
        writer.writerow(["computed_distance_feature", "hypo_dist_km=sqrt(epi_dist_km^2+(dep_km+elev_km)^2)"])
        writer.writerow(["phase_requirement", args.phase_requirement])
        writer.writerow(["max_hypo_dist_km", args.max_hypo_dist_km])
        writer.writerow(["figure_dpi", args.figure_dpi])
        writer.writerow(["station_split_level", args.station_split_level])
        writer.writerow(["validation_fraction", args.validation_fraction])
        writer.writerow(["random_seed", args.random_seed])
        writer.writerow(["rarity_features", "magnitude_FMD;spatiotemporal_seismicity_rate;hypo_dist_km"])
        writer.writerow(["rarity_percentiles", ";".join(map(str, args.rarity_percentiles))])
        writer.writerow(["rarity_thresholds", ";".join(f"{item:.6f}" for item in thresholds)])
        writer.writerow(["augmentation_values", ";".join(map(str, args.augmentation_values))])
        writer.writerow(["validation_num_aug", args.validation_num_aug])
        for key in sorted(counts):
            writer.writerow([key, counts[key]])
        writer.writerow(["split_keys_total", len(split_by_key)])
        writer.writerow(["split_keys_train", sum(1 for value in split_by_key.values() if value == "train")])
        writer.writerow(["split_keys_valid", sum(1 for value in split_by_key.values() if value == "valid")])
        writer.writerow(["samples_train", split_counts["train"]])
        writer.writerow(["samples_valid", split_counts["valid"]])
        writer.writerow(["train_augmented_samples", train_augmented])
        writer.writerow(["valid_augmented_samples", valid_augmented])
        for num_aug in sorted(aug_counts):
            writer.writerow([f"num_aug_{num_aug}_samples", aug_counts[num_aug]])


def plot_features(hist, args):
    if plt is None:
        return
    Path(args.feature_fig_out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax = ax.ravel()

    mag, mag_edges, mag_pdf, _ = hist["mag"]
    centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    ax[0].hist(mag, bins=mag_edges, density=True, color="0.75", edgecolor="0.3", label="hist")
    ax[0].plot(centers, mag_pdf, "r-", lw=1.8, label="PDF")
    ax[0].set(xlabel="Magnitude", ylabel="PDF", title="Magnitude/FMD PDF")
    ax[0].legend()

    hypo, hypo_edges, hypo_pdf, _ = hist["hypo_dist"]
    centers = 0.5 * (hypo_edges[:-1] + hypo_edges[1:])
    ax[1].hist(hypo, bins=hypo_edges, density=True, color="0.75", edgecolor="0.3", label="hist")
    ax[1].plot(centers, hypo_pdf, "r-", lw=1.8, label="PDF")
    ax[1].set(xlabel="Hypocentral distance (km)", ylabel="PDF", title="Hypocentral Distance PDF")
    ax[1].legend()

    st_counts, st_edges, st_pdf, _ = hist["st_count"]
    centers = 0.5 * (st_edges[:-1] + st_edges[1:])
    ax[2].hist(st_counts, bins=st_edges, density=True, color="0.75", edgecolor="0.3", label="hist")
    ax[2].plot(centers, st_pdf, "r-", lw=1.8, label="PDF")
    st_xlabel = (
        f"log10(N); N = events per {args.spatial_bin_km:g} km x "
        f"{args.spatial_bin_km:g} km x {args.spatiotemporal_time_bin_days:g} day bin"
    )
    ax[2].set(xlabel=st_xlabel, ylabel="PDF", title="Spatiotemporal Count Distribution")
    ax[2].legend()

    rate_x = np.asarray(st_counts, dtype=float)
    rate_y = np.asarray(hist["st_density"], dtype=float)
    valid_rate = np.isfinite(rate_x) & np.isfinite(rate_y) & (rate_y > 0)
    ax[3].scatter(rate_x[valid_rate], rate_y[valid_rate], s=4, c="0.65", alpha=0.25, linewidths=0, label="samples")
    rate_curve = np.full_like(centers, np.nan, dtype=float)
    for j in range(len(centers)):
        m = valid_rate & (rate_x >= st_edges[j]) & (rate_x < st_edges[j + 1])
        if np.any(m):
            rate_curve[j] = np.median(rate_y[m])
    good = np.isfinite(rate_curve)
    ax[3].plot(centers[good], rate_curve[good], "r-", lw=1.8, label="p_XYT lookup")
    ax[3].set(
        xlabel=st_xlabel,
        ylabel="p_XYT density 1/(km^2 day)",
        title="Spatiotemporal Rate Used for Rarity",
    )
    if np.any(valid_rate):
        ax[3].set_yscale("log")
    ax[3].legend()

    for axis in ax:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.feature_fig_out, dpi=args.figure_dpi)
    plt.close(fig)


def plot_prob(hist, args):
    if plt is None:
        return
    Path(args.prob_fig_out).parent.mkdir(parents=True, exist_ok=True)
    p_norm = np.asarray(hist["p_norm"], dtype=float)
    rarity = np.asarray(hist["rarity"], dtype=float)
    thresholds = np.asarray(hist["thresholds"], dtype=float)
    labels = [f"aug={value}" for value in args.augmentation_values]

    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    pos = p_norm[p_norm > 0]
    if len(pos):
        ax[0].hist(np.log10(pos), bins=PROB_HIST_BINS, color="0.55", edgecolor="0.2")
        ax[0].set_xlabel("log10(p / max_p)")
    else:
        ax[0].hist(p_norm, bins=PROB_HIST_BINS, color="0.55", edgecolor="0.2")
        ax[0].set_xlabel("p / max_p")
    ax[0].set_ylabel("Station waveform count")
    ax[0].set_title("Normalized Joint Probability Distribution")
    ax[0].grid(True, alpha=0.3)

    xmin = float(np.nanmin(rarity))
    xmax = float(np.nanmax(rarity))
    bounds = np.r_[xmin, thresholds, xmax]
    if bounds[-1] <= bounds[0]:
        bounds[-1] = bounds[0] + 1e-6
    colors = ["#e8f4ff", "#d8ecff", "#c7e3ff", "#b5d9ff", "#a3ceff", "#90c2ff"]
    for i in range(len(bounds) - 1):
        color = colors[i % len(colors)]
        ax[1].axvspan(bounds[i], bounds[i + 1], color=color, alpha=0.55, lw=0)
        xmid = 0.5 * (bounds[i] + bounds[i + 1])
        label = labels[i] if i < len(labels) else f"zone {i + 1}"
        ax[1].text(
            xmid,
            0.96,
            label,
            transform=ax[1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
    ax[1].hist(rarity, bins=PROB_HIST_BINS, color="0.55", edgecolor="0.2", alpha=0.75)
    for pctl, threshold in zip(args.rarity_percentiles, thresholds):
        ax[1].axvline(threshold, color="r", lw=1.2, ls="--")
        ax[1].text(
            threshold,
            0.88,
            f"P{pctl:g}",
            transform=ax[1].get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            color="r",
            fontsize=8,
        )
    ax[1].set_xlabel(f"rarity = -log10(max(p_norm, {RARITY_FLOOR:g}))")
    ax[1].set_ylabel("Station waveform count")
    ax[1].set_title("Rarity Distribution with Augmentation Percentile Zones")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.prob_fig_out, dpi=args.figure_dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Assign rarity-aware augmentation counts to PAL phase picks.")
    parser.add_argument("--phase-in", default="output/ceed_phase.pha")
    parser.add_argument("--station-file", required=True)
    parser.add_argument("--feature-csv-out", default="output/ceed_phase_station_feature_rarity.csv")
    parser.add_argument("--training-csv-out", default="output/ceed_phase_train_augmented.csv")
    parser.add_argument("--training-pha-out", default="output/ceed_phase_train_augmented.pha")
    parser.add_argument("--summary-out", default="output/ceed_phase_train_augmented_summary.csv")
    parser.add_argument("--feature-fig-out", default="output/ceed_phase_feature_distributions.jpg")
    parser.add_argument("--prob-fig-out", default="output/ceed_phase_rarity_distribution.jpg")
    parser.add_argument("--max-hypo-dist-km", type=float, default=200.0)
    parser.add_argument("--augmentation-values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--rarity-percentiles", type=float, nargs="+", default=[50, 75, 90])
    parser.add_argument("--mag-bin-width", type=float, default=0.1)
    parser.add_argument("--hypo-dist-bin-width", type=float, default=10.0)
    parser.add_argument("--spatial-bin-km", type=float, default=5.0)
    parser.add_argument("--spatiotemporal-time-bin-days", type=float, default=10.0)
    parser.set_defaults(
        phase_requirement=PHASE_REQUIREMENT,
        figure_dpi=FIGURE_DPI,
        station_split_level=STATION_SPLIT_LEVEL,
        validation_fraction=VALIDATION_FRACTION,
        random_seed=RANDOM_SEED,
        validation_num_aug=VALIDATION_NUM_AUG,
        mag_hist_smooth_sigma_bins=MAG_SMOOTH_SIGMA_BINS,
        hypo_dist_hist_smooth_sigma_bins=HYPO_DIST_SMOOTH_SIGMA_BINS,
        spatiotemporal_count_log_bin_width=SPATIOTEMPORAL_COUNT_LOG_BIN_WIDTH,
        spatiotemporal_count_hist_smooth_sigma_bins=SPATIOTEMPORAL_COUNT_SMOOTH_SIGMA_BINS,
        spatiotemporal_density_smooth_sigma_xy_bins=SPATIOTEMPORAL_DENSITY_SMOOTH_SIGMA_XY_BINS,
        spatiotemporal_density_smooth_sigma_time_bins=SPATIOTEMPORAL_DENSITY_SMOOTH_SIGMA_TIME_BINS,
        spatiotemporal_density_floor=SPATIOTEMPORAL_DENSITY_FLOOR,
        spatiotemporal_scaled_floor=SPATIOTEMPORAL_SCALED_FLOOR,
    )
    args = parser.parse_args()
    if len(args.augmentation_values) != len(args.rarity_percentiles) + 1:
        raise ValueError("augmentation-values must have one more value than rarity-percentiles")
    return args


def main():
    args = parse_args()
    stations = read_station_coordinates(args.station_file)
    event_headers, samples, counts = read_pal_phase(
        args.phase_in, args.phase_requirement, stations
    )
    samples = filter_samples_by_hypo_distance(samples, args.max_hypo_dist_km, counts)
    split_by_key = assign_station_splits(
        samples,
        args.validation_fraction,
        args.random_seed,
        args.station_split_level,
    )
    thresholds, hist = compute_rarity(samples, args)
    apply_num_aug(samples, args.validation_num_aug)

    write_feature_csv(samples, args.feature_csv_out)
    write_training_csv(samples, args.training_csv_out)
    write_training_pha(event_headers, samples, args.training_pha_out)
    write_summary(args.summary_out, args, counts, split_by_key, thresholds, samples)
    plot_features(hist, args)
    plot_prob(hist, args)

    split_counts = Counter(s["split"] for s in samples)
    print(f"phase input: {args.phase_in}")
    print(f"station input: {args.station_file}")
    print("epicentral distance source: event and station coordinates")
    print("rarity distance feature: hypo_dist_km = sqrt(epi_dist_km^2 + (dep_km + elev_km)^2)")
    print(f"max hypo distance: {args.max_hypo_dist_km:g} km")
    print(f"samples: {len(samples)}")
    print(f"stations/split keys: {len(split_by_key)}")
    print(f"train samples: {split_counts['train']}")
    print(f"valid samples: {split_counts['valid']}")
    print(f"feature CSV: {args.feature_csv_out}")
    print(f"training CSV: {args.training_csv_out}")
    print(f"training PHA: {args.training_pha_out}")
    print(f"summary: {args.summary_out}")
    print(f"feature figure: {args.feature_fig_out}")
    print(f"probability figure: {args.prob_fig_out}")


if __name__ == "__main__":
    main()
