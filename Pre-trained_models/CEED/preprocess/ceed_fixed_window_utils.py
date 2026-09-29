"""CEED fixed-window waveform and output helpers; no SeisBench dependency."""

from collections import Counter

from fractions import Fraction

import shutil

from pathlib import Path

import numpy as np

import pandas as pd

from scipy import signal

TARGET_SAMPLE_RATE = 100.0

WINDOW_LENGTH_SEC = 40.0

P_TARGET_SEC = 10.0

EARLY_P_FALLBACK_START_SEC = 1.0

DATASET_OVERRIDES = {}

FREQMIN = 1.0

FREQMAX = 20.0

TO_FILTER = True

FILTER_CORNERS = 4

TAPER_MAX_PERCENTAGE = 0.05

TAPER_MAX_LENGTH_SEC = 5.0

INTEGRATE_ACCELERATION = True

ACCELERATION_CHANNEL_PREFIXES = ('HN', 'HL', 'HG', 'BN', 'LN', 'AN')

ACCELERATION_UNIT_TOKENS = ('acc', 'acceleration', 'gal', 'm/s/s', 'm/s^2', 'm/s**2', 'cm/s/s', 'cm/s^2')

_WORKER_ARGS = None

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
    if TO_FILTER:
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
