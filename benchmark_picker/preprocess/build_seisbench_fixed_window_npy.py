"""Build fixed-window NPY shards from downloaded SeisBench datasets.

Output shard format matches the CEED positive-picker NPY convention:
    shard.shape = (n_samples, 3, win_npts + 2)
    shard[:, :, 0] = P arrival time relative to output window start, seconds
    shard[:, :, 1] = S arrival time relative to output window start, seconds, or NaN
    shard[:, :, 2:] = preprocessed waveform, 3 channels, 100 Hz

The builder reads /nas/zhouyj/AI_datasets/<dataset>/phase_index.csv and the
native SeisBench waveform files under <dataset>/traces.  It writes one fixed
40 s window per trace by default, with P nominally at 10 s unless the source
window has too little pre-P buffer, in which case it starts at raw t=1 s.
"""
from collections import Counter
from fractions import Fraction
import json
import multiprocessing as mp
import shutil
from pathlib import Path
import tempfile
import traceback

import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd
from scipy import signal


_ORIGINAL_PD_TO_DATETIME = pd.to_datetime


def install_pandas_mixed_datetime_fallback():
    """Allow SeisBench metadata with mixed timestamp precision in workers."""
    if getattr(pd.to_datetime, '_ai_pal_mixed_fallback', False):
        return

    def to_datetime_with_mixed_fallback(*args, **kwargs):
        try:
            return _ORIGINAL_PD_TO_DATETIME(*args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if (
                "doesn't match format" not in message
                and "does not match format" not in message
                and 'time data' not in message
            ):
                raise
            if kwargs.get('format') == 'mixed':
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs['format'] = 'mixed'
            return _ORIGINAL_PD_TO_DATETIME(*args, **retry_kwargs)

    to_datetime_with_mixed_fallback._ai_pal_mixed_fallback = True
    pd.to_datetime = to_datetime_with_mixed_fallback


install_pandas_mixed_datetime_fallback()


# -----------------------------------------------------------------------------
# I/O and defaults
# -----------------------------------------------------------------------------
DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
OUT_SUBDIR = 'npy_shards_fixed40'
CLEAN_OUTPUT = True
SUMMARY_CSV = DATA_ROOT / 'seisbench_fixed_window_npy_summary.csv'

TARGET_SAMPLE_RATE = 100.0
WINDOW_LENGTH_SEC = 40.0
P_TARGET_SEC = 10.0
EARLY_P_FALLBACK_START_SEC = 1.0
DATASET_OVERRIDES = {
    # Example:
    # 'OBST2024': {'window_length_sec': 35.0, 'p_target_sec': 8.0},
}
MAX_DISTANCE_KM = 200.0
MAX_SP_SEC_WITHOUT_DISTANCE = 20.0
SHARD_SIZE = 1024
NUM_WORKERS = 8
TASK_CHUNK_SIZE = 1024
PROGRESS_EVERY_TASKS = 100

FREQMIN = 1.0
FREQMAX = 20.0
FILTER_CORNERS = 4
TAPER_MAX_PERCENTAGE = 0.05
TAPER_MAX_LENGTH_SEC = 5.0
INTEGRATE_ACCELERATION = True

# For metadata rows without explicit distance, S-P is used as a fallback filter.
REQUIRE_P = True

DATASET_CLASS_ALIASES = {
    'INSTANCE': ['InstanceCounts'],
    'CWA': ['CWA'],
    'PNW': ['PNW'],
    'STEAD': ['STEAD'],
    'piSDL': ['piSDL', 'PiSDL', 'PISDL', 'pISDL'],
    'OBST2024': ['OBST2024'],
}

ACCELERATION_CHANNEL_PREFIXES = ('HN', 'HL', 'HG', 'BN', 'LN', 'AN')
ACCELERATION_UNIT_TOKENS = ('acc', 'acceleration', 'gal', 'm/s/s', 'm/s^2', 'm/s**2', 'cm/s/s', 'cm/s^2')


# -----------------------------------------------------------------------------
# Cache control and SeisBench loading
# -----------------------------------------------------------------------------
def configure_large_file_cache():
    cache_root = DATA_ROOT / '_cache'
    tmp_root = DATA_ROOT / '_tmp'
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    env_paths = {
        'SEISBENCH_CACHE_ROOT': cache_root / 'seisbench',
        'XDG_CACHE_HOME': cache_root / 'xdg',
        'POOCH_HOME': cache_root / 'pooch',
        'MPLCONFIGDIR': cache_root / 'matplotlib',
        'NUMBA_CACHE_DIR': cache_root / 'numba',
        'TMPDIR': tmp_root,
        'TEMP': tmp_root,
        'TMP': tmp_root,
    }
    for key, value in env_paths.items():
        value.mkdir(parents=True, exist_ok=True)
        import os
        os.environ[key] = str(value)
    tempfile.tempdir = str(tmp_root)


configure_large_file_cache()


def import_seisbench():
    import seisbench as sb
    import seisbench.data as sbd
    try:
        sb.cache_root = DATA_ROOT / '_cache' / 'seisbench'
    except Exception:
        pass
    return sb, sbd


def resolve_dataset_class(sbd, dataset_name):
    aliases = DATASET_CLASS_ALIASES.get(dataset_name, [dataset_name])
    for alias in aliases:
        cls = getattr(sbd, alias, None)
        if cls is not None:
            return alias, cls
    raise AttributeError('No SeisBench class found for {} aliases {}'.format(dataset_name, aliases))


def construct_dataset(cls, path):
    kwargs = {
        'path': path,
        'metadata_cache': False,
        'missing_components': 'ignore',
        'component_order': 'ENZ',
    }
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop('missing_components', None)
        try:
            return cls(**kwargs)
        except TypeError:
            return cls(path=path)


# Worker globals
_WORKER_DATASET = None
_WORKER_DATASET_NAME = None
_WORKER_OUT_DIR = None
_WORKER_ARGS = None


def init_worker(dataset_name, dataset_path, out_dir, args_dict):
    global _WORKER_DATASET, _WORKER_DATASET_NAME, _WORKER_OUT_DIR, _WORKER_ARGS
    install_pandas_mixed_datetime_fallback()
    _, sbd = import_seisbench()
    _, cls = resolve_dataset_class(sbd, dataset_name)
    _WORKER_DATASET = construct_dataset(cls, Path(dataset_path))
    _WORKER_DATASET_NAME = dataset_name
    _WORKER_OUT_DIR = Path(out_dir)
    _WORKER_ARGS = args_dict



def dataset_settings(dataset_name):
    settings = {
        'window_length_sec': WINDOW_LENGTH_SEC,
        'p_target_sec': P_TARGET_SEC,
        'early_p_fallback_start_sec': EARLY_P_FALLBACK_START_SEC,
    }
    settings.update(DATASET_OVERRIDES.get(dataset_name, {}))
    return settings


def setting(name):
    if _WORKER_ARGS is None:
        return dataset_settings('default')[name]
    return _WORKER_ARGS.get(name, dataset_settings('default')[name])

# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------
def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value, default=''):
    if value is None or pd.isna(value):
        return default
    text = str(value)
    return default if text.lower() == 'nan' else text


def get_first(row, names, default=np.nan):
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def load_phase_index(dataset_name):
    path = DATA_ROOT / dataset_name / 'phase_index.csv'
    if not path.exists():
        print('skip {}: missing {}'.format(dataset_name, path))
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'sb_idx' not in df.columns:
        df['sb_idx'] = np.arange(len(df), dtype=np.int64)
    return df


def channel_is_acceleration(channel, units=''):
    channel = safe_str(channel).upper()
    units = safe_str(units).lower()
    for token in ACCELERATION_UNIT_TOKENS:
        if token in units:
            return True
    if len(channel) >= 2 and channel[:2] in ACCELERATION_CHANNEL_PREFIXES:
        return True
    # Handle comma/pipe-separated channel lists.
    for sep in (',', '|', ';', ' '):
        if sep in channel:
            return any(channel_is_acceleration(part, units='') for part in channel.split(sep))
    return False


def row_to_info(row):
    sr = safe_float(get_first(row, ['sampling_rate_hz', 'trace_sampling_rate_hz', 'sampling_rate']))
    p_sec = safe_float(get_first(row, ['p_arrival_sec']))
    s_sec = safe_float(get_first(row, ['s_arrival_sec']))
    if not np.isfinite(p_sec):
        p_sample = safe_float(get_first(row, ['p_arrival_sample', 'trace_p_arrival_sample', 'trace_P_arrival_sample']))
        if np.isfinite(p_sample) and np.isfinite(sr) and sr > 0:
            p_sec = p_sample / sr
    if not np.isfinite(s_sec):
        s_sample = safe_float(get_first(row, ['s_arrival_sample', 'trace_s_arrival_sample', 'trace_S_arrival_sample']))
        if np.isfinite(s_sample) and np.isfinite(sr) and sr > 0:
            s_sec = s_sample / sr
    dist = safe_float(get_first(row, ['distance_km', 'source_distance_km', 'source_receiver_distance_km']))
    trace_len = safe_float(get_first(row, ['trace_length_sec']))
    if not np.isfinite(trace_len):
        npts = safe_float(get_first(row, ['trace_npts', 'trace_n_samples', 'npts']))
        if np.isfinite(npts) and np.isfinite(sr) and sr > 0:
            trace_len = npts / sr
    return {
        'sb_idx': int(row['sb_idx']),
        'sampling_rate_hz': sr,
        'p_sec': p_sec,
        's_sec': s_sec,
        'distance_km': dist,
        'trace_length_sec': trace_len,
        'station_key': safe_str(row.get('station_key'), ''),
        'trace_name': safe_str(row.get('trace_name'), ''),
        'trace_chunk': safe_str(row.get('trace_chunk'), ''),
        'channel': safe_str(row.get('channel'), ''),
        'units': safe_str(get_first(row, ['trace_units', 'trace_unit', 'units'], default='')),
        'event_id': safe_str(row.get('event_id'), ''),
        'origin_time': safe_str(row.get('origin_time'), ''),
        'source_lat': safe_float(row.get('source_lat', np.nan)),
        'source_lon': safe_float(row.get('source_lon', np.nan)),
        'source_depth_km': safe_float(row.get('source_depth_km', np.nan)),
        'source_mag': safe_float(row.get('source_mag', np.nan)),
        'receiver_lat': safe_float(row.get('receiver_lat', np.nan)),
        'receiver_lon': safe_float(row.get('receiver_lon', np.nan)),
    }


def should_keep_trace(info):
    if REQUIRE_P and not np.isfinite(info['p_sec']):
        return False, 'missing_p'
    if not np.isfinite(info['sampling_rate_hz']) or info['sampling_rate_hz'] <= 0:
        return False, 'missing_sampling_rate'
    if np.isfinite(info['distance_km']):
        if info['distance_km'] > MAX_DISTANCE_KM:
            return False, 'distance_gt_max'
        return True, 'kept'

    # If distance metadata is unavailable, S-P is the fallback scale filter.
    if not np.isfinite(info['s_sec']):
        return False, 'missing_distance_and_s'
    sp = info['s_sec'] - info['p_sec']
    if not np.isfinite(sp) or sp <= 0:
        return False, 'invalid_sp_time'
    if sp > MAX_SP_SEC_WITHOUT_DISTANCE:
        return False, 'sp_gt_max_without_distance'
    return True, 'kept'


# -----------------------------------------------------------------------------
# Waveform preprocessing
# -----------------------------------------------------------------------------
def standardize_waveform_shape(waveform):
    data = np.asarray(waveform, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim == 2:
        if data.shape[0] <= 8 and data.shape[1] > data.shape[0]:
            pass
        elif data.shape[1] <= 8 and data.shape[0] > data.shape[1]:
            data = data.T
        else:
            # Ambiguous, assume first axis is channel as in SeisBench CW order.
            pass
    else:
        # Flatten all leading dimensions except the longest sample axis.
        sample_axis = int(np.argmax(data.shape))
        data = np.moveaxis(data, sample_axis, -1)
        data = data.reshape(-1, data.shape[-1])
    if data.shape[0] == 0 or data.shape[1] == 0:
        return None
    return np.asarray(data, dtype=np.float32)


def force_three_channels(data):
    counts = Counter()
    orig_num_channels = int(data.shape[0])
    channel_mode = 'three_channels'
    channel_indices = '0,1,2'
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
        channel_mode = 'single_repeated_0_0_0'
        channel_indices = '0,0,0'
        counts['single_channel_repeated_to_three'] += 1
    elif data.shape[0] == 2:
        data = np.vstack([data, data[-1:]])
        channel_mode = 'two_channels_repeated_0_1_1'
        channel_indices = '0,1,1'
        counts['two_channels_last_repeated_to_three'] += 1
    elif data.shape[0] >= 3:
        if data.shape[0] > 3:
            counts['more_than_three_channels_truncated'] += 1
            channel_mode = 'first_three_channels_0_1_2'
            channel_indices = '0,1,2'
        data = data[:3]
    return data.astype(np.float32, copy=False), counts, {'orig_num_channels': orig_num_channels, 'channel_mode': channel_mode, 'channel_indices': channel_indices}


def clean_nonfinite(data):
    data = np.asarray(data, dtype=np.float32)
    data[~np.isfinite(data)] = 0.0
    return data


def demean_detrend(data):
    data = clean_nonfinite(data)
    data = data - np.mean(data, axis=1, keepdims=True)
    try:
        data = signal.detrend(data, axis=1, type='linear').astype(np.float32, copy=False)
    except Exception:
        data = data - np.mean(data, axis=1, keepdims=True)
    return data.astype(np.float32, copy=False)


def integrate_acceleration(data, sr):
    data = demean_detrend(data)
    return (np.cumsum(data, axis=1) / float(sr)).astype(np.float32, copy=False)


def resample_to_target(data, sr, target_sr):
    if abs(sr - target_sr) <= 1e-5:
        return data.astype(np.float32, copy=False)
    ratio = Fraction(float(target_sr) / float(sr)).limit_denominator(1000)
    data = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=1)
    return data.astype(np.float32, copy=False)


def taper(data, sr):
    npts = data.shape[1]
    taper_len = int(min(TAPER_MAX_LENGTH_SEC * sr, TAPER_MAX_PERCENTAGE * npts))
    if taper_len <= 1:
        return data
    win = np.ones(npts, dtype=np.float32)
    edge = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_len, dtype=np.float32)))
    win[:taper_len] = edge
    win[-taper_len:] = edge[::-1]
    return (data * win[None, :]).astype(np.float32, copy=False)


def bandpass(data, sr):
    nyq = 0.5 * float(sr)
    high = min(float(FREQMAX), 0.9 * nyq)
    low = float(FREQMIN)
    if not np.isfinite(high) or high <= low or high <= 0:
        return data.astype(np.float32, copy=False)
    sos = signal.butter(FILTER_CORNERS, [low / nyq, high / nyq], btype='bandpass', output='sos')
    try:
        data = signal.sosfiltfilt(sos, data, axis=1)
    except ValueError:
        data = signal.sosfilt(sos, data, axis=1)
    return data.astype(np.float32, copy=False)


def preprocess_waveform(data, info):
    counts = Counter()
    sr = float(info['sampling_rate_hz'])
    data = standardize_waveform_shape(data)
    if data is None:
        counts['bad_waveform_shape'] += 1
        return None, counts, {}
    data, ch_counts, prep_info = force_three_channels(data)
    counts.update(ch_counts)
    data = clean_nonfinite(data)

    if INTEGRATE_ACCELERATION and channel_is_acceleration(info['channel'], info['units']):
        data = integrate_acceleration(data, sr)
        counts['acceleration_integrated'] += 1

    if abs(sr - TARGET_SAMPLE_RATE) > 1e-5:
        data = demean_detrend(data)
        data = resample_to_target(data, sr, TARGET_SAMPLE_RATE)
        counts['resampled_to_target'] += 1

    data = demean_detrend(data)
    data = taper(data, TARGET_SAMPLE_RATE)
    data = bandpass(data, TARGET_SAMPLE_RATE)
    return data.astype(np.float32, copy=False), counts, prep_info


def fixed_window_start(info, data_npts):
    p_sec = info['p_sec']
    p_target = setting('p_target_sec')
    if p_sec >= p_target:
        start = p_sec - p_target
    else:
        start = setting('early_p_fallback_start_sec')
    if start < 0:
        start = 0.0
    return float(start)


def cut_failure(reason, start_sec=np.nan, end_sec=np.nan, raw_end_sec=np.nan):
    return {
        'skip_reason': reason,
        'window_start_sec': float(start_sec) if np.isfinite(start_sec) else np.nan,
        'window_end_sec': float(end_sec) if np.isfinite(end_sec) else np.nan,
        'raw_end_sec': float(raw_end_sec) if np.isfinite(raw_end_sec) else np.nan,
    }


def cut_fixed_window(data, info):
    window_length_sec = setting('window_length_sec')
    win_npts = int(round(window_length_sec * TARGET_SAMPLE_RATE))
    raw_end_sec = data.shape[1] / float(TARGET_SAMPLE_RATE)
    if data.shape[1] < win_npts:
        return None, cut_failure('raw_window_shorter_than_fixed_window', 0.0, window_length_sec, raw_end_sec)
    start_sec = fixed_window_start(info, data.shape[1])
    start_idx = int(round(start_sec * TARGET_SAMPLE_RATE))
    end_idx = start_idx + win_npts
    start_sec = start_idx / float(TARGET_SAMPLE_RATE)
    end_sec = end_idx / float(TARGET_SAMPLE_RATE)
    if start_idx < 0:
        return None, cut_failure('fixed_window_start_before_raw_start', start_sec, end_sec, raw_end_sec)
    if end_idx > data.shape[1]:
        return None, cut_failure('fixed_window_end_after_raw_end', start_sec, end_sec, raw_end_sec)
    p_rel = info['p_sec'] - start_sec
    s_rel = info['s_sec'] - start_sec if np.isfinite(info['s_sec']) else np.nan
    if p_rel < 0 or p_rel > window_length_sec:
        details = cut_failure('p_outside_fixed_window', start_sec, end_sec, raw_end_sec)
        details['p_rel_sec'] = float(p_rel)
        return None, details
    if np.isfinite(s_rel) and (s_rel < 0 or s_rel > window_length_sec):
        s_rel = np.nan
    out = np.zeros((3, win_npts + 2), dtype=np.float32)
    out[:, 0] = float(p_rel)
    out[:, 1] = float(s_rel) if np.isfinite(s_rel) else np.nan
    out[:, 2:] = data[:, start_idx:end_idx]
    return out, {
        'window_start_sec': start_sec,
        'window_end_sec': end_sec,
        'raw_end_sec': raw_end_sec,
        'p_rel_sec': float(p_rel),
        's_rel_sec': float(s_rel) if np.isfinite(s_rel) else np.nan,
        'has_s_in_window': bool(np.isfinite(s_rel)),
    }


def normalize_channels(sample):
    data = sample[:, 2:]
    data = data - np.mean(data, axis=1, keepdims=True)
    scale = np.max(np.abs(data), axis=1, keepdims=True)
    scale[scale <= 0] = 1.0
    sample[:, 2:] = data / scale
    return sample.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Worker and shard writing
# -----------------------------------------------------------------------------
def get_waveform(ds, sb_idx):
    if hasattr(ds, 'get_waveforms'):
        return ds.get_waveforms(int(sb_idx))
    sample = ds.get_sample(int(sb_idx))
    if isinstance(sample, tuple):
        return sample[0]
    if isinstance(sample, dict):
        for key in ('X', 'waveforms', 'data'):
            if key in sample:
                return sample[key]
    return sample


def failure_meta(info, reason, extra=None):
    meta = dict(info)
    meta['dataset'] = _WORKER_DATASET_NAME
    meta['skip_reason'] = reason
    if extra:
        meta.update(extra)
    return meta


def process_one_record(row):
    counts = Counter()
    info = row_to_info(row)
    keep, reason = should_keep_trace(info)
    if not keep:
        counts['skipped_' + reason] += 1
        return None, None, failure_meta(info, reason), counts
    try:
        waveform = get_waveform(_WORKER_DATASET, info['sb_idx'])
        data, prep_counts, prep_info = preprocess_waveform(waveform, info)
        counts.update(prep_counts)
        if data is None:
            counts['skipped_bad_waveform'] += 1
            return None, None, failure_meta(info, 'bad_waveform'), counts
        sample, cut_info = cut_fixed_window(data, info)
        if sample is None:
            reason = cut_info.get('skip_reason', 'fixed_window_failed')
            counts['skipped_' + reason] += 1
            return None, None, failure_meta(info, reason, cut_info), counts
        sample = normalize_channels(sample)
        counts['samples_written'] += 1
        meta = dict(info)
        meta.update(prep_info)
        meta.update(cut_info)
        meta['dataset'] = _WORKER_DATASET_NAME
        return sample, meta, None, counts
    except Exception as exc:
        counts['skipped_exception'] += 1
        return None, None, failure_meta(info, 'exception', {'error': repr(exc)}), counts


def process_task(task):
    task_idx, rows = task
    counts = Counter()
    samples = []
    metas = []
    failures = []
    for row in rows:
        sample, meta, failure, row_counts = process_one_record(row)
        counts.update(row_counts)
        if sample is not None:
            samples.append(sample)
            metas.append(meta)
        elif failure is not None:
            failures.append(failure)

    shard_rows = []
    if samples:
        shard_dir = _WORKER_OUT_DIR / 'shards'
        index_dir = _WORKER_OUT_DIR / 'index_parts'
        shard_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        shard = np.stack(samples, axis=0).astype(np.float32, copy=False)
        shard_path = shard_dir / ('%s_shard_%06d.npy' % (_WORKER_DATASET_NAME, task_idx))
        np.save(shard_path, shard)
        shard_rows.append((str(shard_path), str(shard.shape[0])))
        meta_df = pd.DataFrame(metas)
        meta_df['shard_path'] = str(shard_path)
        meta_df['row_in_shard'] = np.arange(len(meta_df), dtype=np.int64)
        meta_df.to_csv(index_dir / ('part_%06d.csv' % task_idx), index=False)
    if failures:
        failure_dir = _WORKER_OUT_DIR / 'failure_parts'
        failure_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_dir / ('part_%06d.csv' % task_idx), index=False)
    counts['tasks_done'] += 1
    return counts, shard_rows


def iter_tasks(df, chunk_size):
    task_idx = 0
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        yield task_idx, chunk.to_dict(orient='records')
        task_idx += 1


def merge_csv_parts(parts_dir, out_path):
    parts = sorted(parts_dir.glob('part_*.csv'))
    if not parts:
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path
    with open(out_path, 'w') as fout:
        wrote_header = False
        for part in parts:
            with open(part) as fin:
                header = fin.readline()
                if not wrote_header:
                    fout.write(header)
                    wrote_header = True
                for line in fin:
                    fout.write(line)
    return out_path


def merge_index_parts(out_dir):
    return merge_csv_parts(out_dir / 'index_parts', out_dir / 'sample_index.csv')


def merge_failure_parts(out_dir):
    return merge_csv_parts(out_dir / 'failure_parts', out_dir / 'failure_index.csv')


def write_cleaned_phase_file(path, sample_index_path):
    columns = [
        'origin_time', 'source_lat', 'source_lon', 'source_depth_km', 'source_mag',
        'event_id', 'station_key', 'p_rel_sec', 's_rel_sec', 'distance_km',
        'sb_idx', 'trace_name', 'window_start_sec', 'shard_path', 'row_in_shard'
    ]
    header = list(pd.read_csv(sample_index_path, nrows=0).columns)
    usecols = [col for col in columns if col in header]
    with open(path, 'w') as fp:
        if not usecols:
            return path
        for chunk in pd.read_csv(sample_index_path, usecols=usecols, chunksize=200000):
            for _, row in chunk.iterrows():
                event_id = safe_str(row.get('event_id'), '')
                if not event_id:
                    event_id = '{}:{}'.format(safe_str(row.get('sb_idx'), ''), safe_str(row.get('trace_name'), ''))
                fp.write('{},{},{},{},{},{}\n'.format(
                    safe_str(row.get('origin_time'), 'NaT'),
                    safe_float(row.get('source_lat', np.nan)),
                    safe_float(row.get('source_lon', np.nan)),
                    safe_float(row.get('source_depth_km', np.nan)),
                    safe_float(row.get('source_mag', np.nan)),
                    event_id,
                ))
                fp.write('{},{},{},{},{},{},{},{},{}\n'.format(
                    safe_str(row.get('station_key'), '--.--..'),
                    safe_float(row.get('p_rel_sec', np.nan)),
                    safe_float(row.get('s_rel_sec', np.nan)),
                    safe_float(row.get('distance_km', np.nan)),
                    int(safe_float(row.get('sb_idx', -1), -1)),
                    safe_str(row.get('trace_name'), ''),
                    safe_float(row.get('window_start_sec', np.nan)),
                    safe_str(row.get('shard_path'), ''),
                    int(safe_float(row.get('row_in_shard', -1), -1)),
                ))
    return path


def write_cleaned_outputs(out_dir, sample_index_path):
    cleaned_index_path = out_dir / 'cleaned_phase_index.csv'
    if Path(sample_index_path).exists():
        shutil.copyfile(sample_index_path, cleaned_index_path)
    else:
        pd.DataFrame().to_csv(cleaned_index_path, index=False)
    cleaned_phase_path = out_dir / 'cleaned_phase.pha'
    write_cleaned_phase_file(cleaned_phase_path, cleaned_index_path)
    return cleaned_index_path, cleaned_phase_path


def reset_output_dir(out_dir):
    if not CLEAN_OUTPUT:
        out_dir.mkdir(parents=True, exist_ok=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ['shards', 'index_parts', 'failure_parts']:
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in [
        'pos.npy',
        'sample_index.csv',
        'failure_index.csv',
        'cleaned_phase_index.csv',
        'cleaned_phase.pha',
        'summary.json',
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def write_shard_index(out_dir, shard_rows):
    arr = np.asarray(sorted(shard_rows, key=lambda item: item[0]), dtype=str)
    path = out_dir / 'pos.npy'
    np.save(path, arr)
    return path


def process_dataset(dataset_name):
    df = load_phase_index(dataset_name)
    if df.empty:
        return {'dataset': dataset_name, 'status': 'missing_index'}
    dataset_path = DATA_ROOT / dataset_name / 'traces'
    out_dir = DATA_ROOT / dataset_name / OUT_SUBDIR
    reset_output_dir(out_dir)
    print('=' * 80, flush=True)
    print('building {} -> {}'.format(dataset_name, out_dir), flush=True)
    print('metadata rows: {:,}'.format(len(df)), flush=True)

    args_dict = dataset_settings(dataset_name)
    counts = Counter()
    shard_rows = []
    tasks = iter_tasks(df, TASK_CHUNK_SIZE)
    if NUM_WORKERS <= 1:
        init_worker(dataset_name, dataset_path, out_dir, args_dict)
        for task_idx, task in enumerate(tasks, start=1):
            task_counts, task_rows = process_task(task)
            counts.update(task_counts)
            shard_rows.extend(task_rows)
            if task_idx % PROGRESS_EVERY_TASKS == 0:
                print('{}: {} tasks | samples {:,}'.format(dataset_name, task_idx, counts['samples_written']), flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(
            processes=NUM_WORKERS,
            initializer=init_worker,
            initargs=(dataset_name, str(dataset_path), str(out_dir), args_dict),
        ) as pool:
            for task_idx, (task_counts, task_rows) in enumerate(pool.imap_unordered(process_task, tasks), start=1):
                counts.update(task_counts)
                shard_rows.extend(task_rows)
                if task_idx % PROGRESS_EVERY_TASKS == 0:
                    print('{}: {} tasks | samples {:,}'.format(dataset_name, task_idx, counts['samples_written']), flush=True)

    shard_index_path = write_shard_index(out_dir, shard_rows)
    sample_index_path = merge_index_parts(out_dir)
    failure_index_path = merge_failure_parts(out_dir)
    cleaned_index_path, cleaned_phase_path = write_cleaned_outputs(out_dir, sample_index_path)
    summary_path = out_dir / 'summary.json'
    summary = {
        'dataset': dataset_name,
        'status': 'ok',
        'input_phase_index': str(DATA_ROOT / dataset_name / 'phase_index.csv'),
        'dataset_path': str(dataset_path),
        'out_dir': str(out_dir),
        'shard_index': str(shard_index_path),
        'sample_index': str(sample_index_path),
        'failure_index': str(failure_index_path),
        'cleaned_phase_index': str(cleaned_index_path),
        'cleaned_phase': str(cleaned_phase_path),
        'window_length_sec': args_dict['window_length_sec'],
        'target_sample_rate': TARGET_SAMPLE_RATE,
        'p_target_sec': args_dict['p_target_sec'],
        'max_distance_km': MAX_DISTANCE_KM,
        'max_sp_sec_without_distance': MAX_SP_SEC_WITHOUT_DISTANCE,
        'require_p_only': REQUIRE_P,
        'num_input_rows': int(len(df)),
        'num_shards': int(len(shard_rows)),
        'counts': {key: int(val) for key, val in counts.items() if isinstance(val, (int, np.integer))},
    }
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print('{} done | samples {:,} | shards {:,}'.format(dataset_name, counts['samples_written'], len(shard_rows)), flush=True)
    return {
        'dataset': dataset_name,
        'status': 'ok',
        'num_input_rows': int(len(df)),
        'num_samples': int(counts['samples_written']),
        'num_shards': int(len(shard_rows)),
        'out_dir': str(out_dir),
        'shard_index': str(shard_index_path),
        'sample_index': str(sample_index_path),
        'failure_index': str(failure_index_path),
        'cleaned_phase_index': str(cleaned_index_path),
        'cleaned_phase': str(cleaned_phase_path),
    }


def main():
    rows = []
    for dataset_name in DATASETS:
        try:
            rows.append(process_dataset(dataset_name))
        except Exception as exc:
            traceback.print_exc()
            rows.append({
                'dataset': dataset_name,
                'status': 'error',
                'error': repr(exc),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print('summary: {}'.format(SUMMARY_CSV), flush=True)
    print(summary, flush=True)


if __name__ == '__main__':
    main()