"""Build fixed-window NPY shards from raw CEED HDF5 data for reference inference.

This is the CEED counterpart of build_seisbench_fixed_window_npy.py.  It reads
/nas/zhouyj/CEED/quakeflow_{nc,sc}/waveform_h5/*.h5 directly, extracts paired
P/S picks from CEED station dataset attributes, applies the same preprocessing
and fixed-window slicing, and writes CEED reference NPY shards in the same
format as the SeisBench benchmark shards.

Output shard format:
    shard.shape = (n_samples, 3, win_npts + 2)
    shard[:, :, 0] = P arrival time relative to output window start, seconds
    shard[:, :, 1] = S arrival time relative to output window start, seconds
    shard[:, :, 2:] = preprocessed waveform, 3 channels, 100 Hz
"""
from collections import Counter
import csv
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd

import build_seisbench_fixed_window_npy as fixed


# -----------------------------------------------------------------------------
# I/O and defaults
# -----------------------------------------------------------------------------
CEED_ROOT = Path('/nas/zhouyj/CEED')
OUT_ROOT = Path('/nas/zhouyj/AI_datasets/CEED')
OUT_SUBDIR = 'npy_shards_fixed50'
SUMMARY_CSV = OUT_ROOT / 'ceed_fixed_window_npy_summary.csv'

TARGET_SAMPLE_RATE = fixed.TARGET_SAMPLE_RATE
WINDOW_LENGTH_SEC = 50.0
P_TARGET_SEC = 10.0
EARLY_P_FALLBACK_START_SEC = 1.0
MAX_DISTANCE_KM = 200.0
MAX_SP_SEC_WITHOUT_DISTANCE = 20.0
SHARD_SIZE = 1024
NUM_WORKERS = 8
PROGRESS_EVERY_EVENTS = 5000
CLEAN_OUTPUT = True

DATASET_SETTINGS = {
    'window_length_sec': WINDOW_LENGTH_SEC,
    'p_target_sec': P_TARGET_SEC,
    'early_p_fallback_start_sec': EARLY_P_FALLBACK_START_SEC,
}

MISSING_PICK_VALUES = {'', '-1', 'nan', 'NaN', 'None', 'none'}


# -----------------------------------------------------------------------------
# Cache/temp redirection
# -----------------------------------------------------------------------------
def configure_large_file_cache():
    cache_root = OUT_ROOT / '_cache'
    tmp_root = OUT_ROOT / '_tmp'
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    env_paths = {
        'XDG_CACHE_HOME': cache_root / 'xdg',
        'MPLCONFIGDIR': cache_root / 'matplotlib',
        'NUMBA_CACHE_DIR': cache_root / 'numba',
        'TMPDIR': tmp_root,
        'TEMP': tmp_root,
        'TMP': tmp_root,
    }
    import os
    for key, value in env_paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    tempfile.tempdir = str(tmp_root)


configure_large_file_cache()



def prepare_output_dir(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not CLEAN_OUTPUT:
        return
    for child in ('shards', 'index_parts', 'failure_parts'):
        path = out_dir / child
        if path.exists():
            shutil.rmtree(path)
    for child in ('pos.npy', 'sample_index.csv', 'failure_index.csv', 'cleaned_phase_index.csv', 'cleaned_phase.pha', 'summary.json'):
        path = out_dir / child
        if path.exists():
            path.unlink()

# -----------------------------------------------------------------------------
# CEED metadata helpers
# -----------------------------------------------------------------------------
def decode_value(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def attr_scalar(attrs, name, default=None):
    if name not in attrs:
        return default
    value = attrs[name]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    return decode_value(value)


def attr_list(attrs, name):
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
    if text in MISSING_PICK_VALUES:
        return None
    if text.endswith('Z'):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def datetime_to_seconds(value):
    return value.toordinal() * 86400.0 + value.hour * 3600.0 + value.minute * 60.0 + value.second + value.microsecond / 1e6


def split_station_key(station_key):
    parts = str(station_key).split('.')
    net = parts[0] if len(parts) > 0 else ''
    sta = parts[1] if len(parts) > 1 else ''
    loc = parts[2] if len(parts) > 2 else ''
    band = parts[3] if len(parts) > 3 else (parts[-1] if parts else '')
    return net, sta, loc, band


def matching_pick_indices(attrs, current_event_id):
    phase_count = len(attr_list(attrs, 'phase_time'))
    pick_event_ids = attr_list(attrs, 'event_id')
    if not pick_event_ids:
        return set(range(phase_count))
    current = str(current_event_id)
    if len(pick_event_ids) == 1 and phase_count > 1:
        return set(range(phase_count)) if str(pick_event_ids[0]) == current else set()
    return {idx for idx, event_id in enumerate(pick_event_ids) if str(event_id) == current}


def paired_pick_times(station_dataset, event_id):
    attrs = station_dataset.attrs
    phase_types = attr_list(attrs, 'phase_type')
    phase_times = attr_list(attrs, 'phase_time')
    keep = matching_pick_indices(attrs, event_id)
    picks = {'P': [], 'S': []}
    for idx, phase_type in enumerate(phase_types):
        if idx not in keep or idx >= len(phase_times):
            continue
        phase = str(phase_type).strip().upper()[:1]
        if phase not in picks:
            continue
        t = parse_time(phase_times[idx])
        if t is not None:
            picks[phase].append(t)
    if not picks['P'] or not picks['S']:
        return None, None
    return min(picks['P']), min(picks['S'])


def ceed_h5_files():
    roots = [
        CEED_ROOT / 'quakeflow_nc' / 'waveform_h5',
        CEED_ROOT / 'quakeflow_sc' / 'waveform_h5',
    ]
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.glob('*.h5')))
        else:
            print('missing CEED waveform root: {}'.format(root), flush=True)
    return sorted(files)


def event_origin_info(event_group, event_id):
    attrs = event_group.attrs
    ot = parse_time(attr_scalar(attrs, 'event_time'))
    begin = parse_time(attr_scalar(attrs, 'begin_time'))
    sr = fixed.safe_float(attr_scalar(attrs, 'sampling_rate', 100.0))
    return {
        'event_id': str(event_id),
        'origin_time': ot.isoformat(timespec='microseconds') + 'Z' if ot else '',
        'begin_time': begin,
        'sampling_rate_hz': sr,
        'source_lat': fixed.safe_float(attr_scalar(attrs, 'latitude')),
        'source_lon': fixed.safe_float(attr_scalar(attrs, 'longitude')),
        'source_depth_km': fixed.safe_float(attr_scalar(attrs, 'depth_km')),
        'source_mag': fixed.safe_float(attr_scalar(attrs, 'magnitude')),
    }


def station_info(station_key, station_dataset, event_info, tp, ts):
    attrs = station_dataset.attrs
    net, sta, loc, band = split_station_key(station_key)
    begin = event_info['begin_time']
    if begin is None:
        return None, 'missing_begin_time'
    p_sec = datetime_to_seconds(tp) - datetime_to_seconds(begin)
    s_sec = datetime_to_seconds(ts) - datetime_to_seconds(begin)
    dist = fixed.safe_float(attr_scalar(attrs, 'distance_km'))
    if np.isfinite(dist) and dist > MAX_DISTANCE_KM:
        return None, 'distance_gt_max'
    sp = s_sec - p_sec
    if (not np.isfinite(dist)) and sp > MAX_SP_SEC_WITHOUT_DISTANCE:
        return None, 'sp_gt_max_without_distance'
    if not np.isfinite(sp) or sp <= 0:
        return None, 'invalid_sp_time'
    return {
        'dataset': 'CEED',
        'h5_path': '',
        'event_id': event_info['event_id'],
        'origin_time': event_info['origin_time'],
        'station_key': station_key,
        'network': net,
        'station': sta,
        'location': loc,
        'channel': band,
        'sampling_rate_hz': event_info['sampling_rate_hz'],
        'p_sec': float(p_sec),
        's_sec': float(s_sec),
        'distance_km': float(dist) if np.isfinite(dist) else np.nan,
        'source_lat': event_info['source_lat'],
        'source_lon': event_info['source_lon'],
        'source_depth_km': event_info['source_depth_km'],
        'source_mag': event_info['source_mag'],
        'receiver_lat': fixed.safe_float(attr_scalar(attrs, 'latitude')),
        'receiver_lon': fixed.safe_float(attr_scalar(attrs, 'longitude')),
        'receiver_elev_m': fixed.safe_float(attr_scalar(attrs, 'elevation_m')),
        'trace_length_sec': np.nan,
        'trace_name': station_key,
        'trace_chunk': '',
        'units': '',
    }, 'kept'


# -----------------------------------------------------------------------------
# Waveform conversion and shard writing
# -----------------------------------------------------------------------------
def failure_meta(info, reason, extra=None):
    meta = dict(info)
    meta['dataset'] = 'CEED'
    meta['skip_reason'] = reason
    if extra:
        meta.update(extra)
    return meta


def make_sample_from_dataset(station_dataset, info):
    info = dict(info)
    waveform = np.asarray(station_dataset[:, :], dtype=np.float32)
    if waveform.ndim != 2:
        counts = Counter({'skipped_bad_waveform_shape': 1})
        return None, None, failure_meta(info, 'bad_waveform_shape'), counts
    if waveform.shape[0] != 3 and waveform.shape[1] == 3:
        waveform = waveform.T
    info['trace_length_sec'] = waveform.shape[1] / float(info['sampling_rate_hz'])

    data, prep_counts, prep_info = fixed.preprocess_waveform(waveform, info)
    counts = Counter(prep_counts)
    if data is None:
        counts['skipped_bad_waveform'] += 1
        return None, None, failure_meta(info, 'bad_waveform'), counts

    fixed._WORKER_ARGS = DATASET_SETTINGS
    sample, cut_info = fixed.cut_fixed_window(data, info)
    if sample is None:
        reason = cut_info.get('skip_reason', 'fixed_window_failed') if isinstance(cut_info, dict) else str(cut_info)
        counts['skipped_' + reason] += 1
        extra = cut_info if isinstance(cut_info, dict) else {}
        return None, None, failure_meta(info, reason, extra), counts
    sample = fixed.normalize_channels(sample)
    meta = dict(info)
    meta.update(prep_info)
    meta.update(cut_info)
    counts['samples_written'] += 1
    return sample, meta, None, counts


def write_shard(out_dir, file_index, shard_index, samples, metas):
    shard_dir = out_dir / 'shards'
    index_dir = out_dir / 'index_parts'
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    shard = np.stack(samples, axis=0).astype(np.float32, copy=False)
    shard_path = shard_dir / ('CEED_file_%05d_shard_%06d.npy' % (file_index, shard_index))
    np.save(shard_path, shard)
    meta_df = pd.DataFrame(metas)
    meta_df['shard_path'] = str(shard_path)
    meta_df['row_in_shard'] = np.arange(len(meta_df), dtype=np.int64)
    meta_df.to_csv(index_dir / ('part_%05d_%06d.csv' % (file_index, shard_index)), index=False)
    return str(shard_path), str(shard.shape[0])


def write_failure_part(out_dir, file_index, part_index, failures):
    if not failures:
        return
    failure_dir = out_dir / 'failure_parts'
    failure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(failure_dir / ('part_%05d_%06d.csv' % (file_index, part_index)), index=False)


def process_h5_file(task):
    file_index, h5_path, out_dir = task
    counts = Counter()
    shard_rows = []
    buffer_samples = []
    buffer_metas = []
    buffer_failures = []
    shard_index = 0
    failure_part_index = 0
    print('reading {}'.format(h5_path), flush=True)
    with h5py.File(h5_path, 'r') as h5:
        for event_idx, event_id in enumerate(sorted(h5.keys())):
            event = h5[event_id]
            if not isinstance(event, h5py.Group):
                continue
            event_info = event_origin_info(event, event_id)
            if event_info['begin_time'] is None:
                counts['events_missing_begin_time'] += 1
                continue
            for station_key in sorted(event.keys()):
                obj = event[station_key]
                if not isinstance(obj, h5py.Dataset):
                    continue
                tp, ts = paired_pick_times(obj, event_id)
                if tp is None or ts is None:
                    counts['skipped_missing_p_or_s'] += 1
                    continue
                info, reason = station_info(station_key, obj, event_info, tp, ts)
                if info is None:
                    counts['skipped_' + reason] += 1
                    continue
                info['h5_path'] = str(h5_path)
                info['h5_event_id'] = str(event_id)
                sample, meta, failure, sample_counts = make_sample_from_dataset(obj, info)
                counts.update(sample_counts)
                if sample is None:
                    if failure is not None:
                        buffer_failures.append(failure)
                        if len(buffer_failures) >= SHARD_SIZE:
                            write_failure_part(out_dir, file_index, failure_part_index, buffer_failures)
                            failure_part_index += 1
                            buffer_failures = []
                    continue
                buffer_samples.append(sample)
                buffer_metas.append(meta)
                if len(buffer_samples) >= SHARD_SIZE:
                    shard_rows.append(write_shard(out_dir, file_index, shard_index, buffer_samples, buffer_metas))
                    shard_index += 1
                    buffer_samples = []
                    buffer_metas = []
            counts['events_processed'] += 1
            if PROGRESS_EVERY_EVENTS and counts['events_processed'] % PROGRESS_EVERY_EVENTS == 0:
                print('{}: events {:,} | samples {:,}'.format(h5_path, counts['events_processed'], counts['samples_written']), flush=True)

    if buffer_samples:
        shard_rows.append(write_shard(out_dir, file_index, shard_index, buffer_samples, buffer_metas))
    if buffer_failures:
        write_failure_part(out_dir, file_index, failure_part_index, buffer_failures)
    counts['h5_files_done'] += 1
    counts['npy_shards_written'] += len(shard_rows)
    return counts, shard_rows


def merge_index_parts(out_dir):
    return fixed.merge_csv_parts(out_dir / 'index_parts', out_dir / 'sample_index.csv')


def merge_failure_parts(out_dir):
    return fixed.merge_csv_parts(out_dir / 'failure_parts', out_dir / 'failure_index.csv')


def write_outputs(out_dir, shard_rows, counts, files):
    shard_rows = sorted(shard_rows, key=lambda item: item[0])
    shard_index = out_dir / 'pos.npy'
    np.save(shard_index, np.asarray(shard_rows, dtype=str))
    sample_index = merge_index_parts(out_dir)
    failure_index = merge_failure_parts(out_dir)
    cleaned_phase_index, cleaned_phase = fixed.write_cleaned_outputs(out_dir, sample_index)
    summary = {
        'dataset': 'CEED',
        'ceed_root': str(CEED_ROOT),
        'out_dir': str(out_dir),
        'shard_index': str(shard_index),
        'sample_index': str(sample_index),
        'failure_index': str(failure_index),
        'cleaned_phase_index': str(cleaned_phase_index),
        'cleaned_phase': str(cleaned_phase),
        'window_length_sec': WINDOW_LENGTH_SEC,
        'target_sample_rate': TARGET_SAMPLE_RATE,
        'p_target_sec': P_TARGET_SEC,
        'max_distance_km': MAX_DISTANCE_KM,
        'max_sp_sec_without_distance': MAX_SP_SEC_WITHOUT_DISTANCE,
        'num_h5_files': len(files),
        'num_shards': len(shard_rows),
        'clean_output': CLEAN_OUTPUT,
        'counts': {key: int(val) for key, val in counts.items()},
    }
    with open(out_dir / 'summary.json', 'w') as fp:
        json.dump(summary, fp, indent=2)
    with open(SUMMARY_CSV, 'w', newline='') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(['parameter', 'value'])
        writer.writerow(['ceed_root', CEED_ROOT])
        writer.writerow(['out_dir', out_dir])
        writer.writerow(['shard_index', shard_index])
        writer.writerow(['sample_index', sample_index])
        writer.writerow(['failure_index', failure_index])
        writer.writerow(['cleaned_phase_index', cleaned_phase_index])
        writer.writerow(['cleaned_phase', cleaned_phase])
        writer.writerow(['num_h5_files', len(files)])
        writer.writerow(['num_shards', len(shard_rows)])
        writer.writerow(['clean_output', CLEAN_OUTPUT])
        for key in sorted(counts):
            writer.writerow([key, counts[key]])
    return shard_index, sample_index, failure_index, cleaned_phase_index, cleaned_phase


def main():
    out_dir = OUT_ROOT / OUT_SUBDIR
    prepare_output_dir(out_dir)
    files = ceed_h5_files()
    if not files:
        raise RuntimeError('No CEED HDF5 files found under {}'.format(CEED_ROOT))
    print('CEED HDF5 files: {}'.format(len(files)), flush=True)
    print('output: {}'.format(out_dir), flush=True)

    tasks = [(idx, str(path), out_dir) for idx, path in enumerate(files)]
    counts = Counter()
    shard_rows = []
    if NUM_WORKERS <= 1:
        results = [process_h5_file(task) for task in tasks]
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=NUM_WORKERS) as pool:
            results = list(pool.imap_unordered(process_h5_file, tasks))
    for file_counts, file_rows in results:
        counts.update(file_counts)
        shard_rows.extend(file_rows)

    shard_index, sample_index, failure_index, cleaned_phase_index, cleaned_phase = write_outputs(out_dir, shard_rows, counts, files)
    print('samples written: {:,}'.format(counts['samples_written']), flush=True)
    print('shards written: {:,}'.format(len(shard_rows)), flush=True)
    print('shard index: {}'.format(shard_index), flush=True)
    print('sample index: {}'.format(sample_index), flush=True)
    print('failure index: {}'.format(failure_index), flush=True)
    print('cleaned phase index: {}'.format(cleaned_phase_index), flush=True)
    print('cleaned phase: {}'.format(cleaned_phase), flush=True)
    print('summary: {}'.format(SUMMARY_CSV), flush=True)


if __name__ == '__main__':
    main()