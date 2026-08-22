"""Plot available fixed-window length after P-based start selection.

This diagnostic uses only SeisBench phase_index.csv metadata.  It applies the
same non-window quality checks as build_seisbench_fixed_window_npy.py:

  - P arrival and sampling rate are required
  - if epicentral/source-receiver distance exists, require distance <= 200 km
  - if distance is unavailable, require valid S and S-P <= 20 s
  - trace length must be known to estimate available window length

For rows passing those checks, available length is computed from the intended
slice start:

  start = P - 10 s, if P >= 10 s
  start = raw_start + 1 s, if P < 10 s
  available_len = raw_trace_length_sec - start

Outputs are written under /nas/zhouyj/AI_datasets/figures_available_window_len.
"""
import os
from pathlib import Path
import tempfile

import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd
import matplotlib.pyplot as plt


DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
OUT_DIR = DATA_ROOT / 'figures_available_window_len'
VALUES_CSV = OUT_DIR / 'seisbench_available_window_length_values.csv'
SUMMARY_CSV = OUT_DIR / 'seisbench_available_window_length_summary.csv'
FIG_COMBINED = OUT_DIR / 'seisbench_available_window_length_combined.jpg'
DPI = 600

P_TARGET_SEC = 10.0
EARLY_P_FALLBACK_START_SEC = 1.0
MAX_DISTANCE_KM = 200.0
MAX_SP_SEC_WITHOUT_DISTANCE = 20.0
THRESHOLDS_SEC = [25.0, 30.0, 40.0, 50.0]
BINS = np.arange(0.0, 130.0, 2.0)
WAVEFORM_LENGTH_PROGRESS_EVERY = 100000

DATASET_CLASS_ALIASES = {
    'INSTANCE': ['InstanceCounts'],
    'CWA': ['CWA'],
    'PNW': ['PNW'],
    'STEAD': ['STEAD'],
    'piSDL': ['piSDL', 'PiSDL', 'PISDL', 'pISDL'],
    'OBST2024': ['OBST2024'],
}



def configure_cache_env():
    cache_root = DATA_ROOT / '_cache'
    tmp_root = DATA_ROOT / '_tmp'
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
        os.environ[key] = str(value)
    tempfile.tempdir = str(tmp_root)


def install_pandas_mixed_datetime_fallback():
    original_to_datetime = pd.to_datetime

    def to_datetime_with_mixed_fallback(*args, **kwargs):
        try:
            return original_to_datetime(*args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if (
                "doesn't match format" not in message
                and 'does not match format' not in message
                and 'time data' not in message
            ):
                raise
            if kwargs.get('format') == 'mixed':
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs['format'] = 'mixed'
            return original_to_datetime(*args, **retry_kwargs)

    pd.to_datetime = to_datetime_with_mixed_fallback


def import_seisbench():
    configure_cache_env()
    install_pandas_mixed_datetime_fallback()
    import seisbench as sb
    import seisbench.data as sbd
    try:
        sb.cache_root = DATA_ROOT / '_cache' / 'seisbench'
    except Exception:
        pass
    return sb, sbd


def resolve_dataset_class(sbd, dataset):
    for alias in DATASET_CLASS_ALIASES.get(dataset, [dataset]):
        cls = getattr(sbd, alias, None)
        if cls is not None:
            return alias, cls
    raise AttributeError('No SeisBench class found for {}'.format(dataset))


def construct_dataset(cls, path):
    kwargs = {
        'path': path,
        'metadata_cache': False,
        'missing_components': 'pad',
        'component_order': 'ENZ',
    }
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop('missing_components', None)
        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop('component_order', None)
            return cls(**kwargs)


def get_waveform(ds, index):
    if hasattr(ds, 'get_waveforms'):
        return ds.get_waveforms(int(index))
    sample = ds.get_sample(int(index))
    if isinstance(sample, tuple):
        return sample[0]
    if isinstance(sample, dict):
        for key in ('X', 'waveforms', 'data'):
            if key in sample:
                return sample[key]
    return sample


def waveform_npts(waveform):
    arr = np.asarray(waveform)
    if arr.size == 0:
        return np.nan
    return float(max(arr.shape))

def safe_num(series):
    return pd.to_numeric(series, errors='coerce')


def first_col(df, names, default=np.nan):
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def phase_seconds(df, phase):
    sec = safe_num(first_col(df, ['{}_arrival_sec'.format(phase)]))
    sample = safe_num(first_col(df, [
        '{}_arrival_sample'.format(phase),
        'trace_{}_arrival_sample'.format(phase),
        'trace_{}_arrival_sample'.format(phase.upper()),
    ]))
    sr = sampling_rate(df)
    fill = ~np.isfinite(sec) & np.isfinite(sample) & np.isfinite(sr) & (sr > 0)
    sec = sec.copy()
    sec.loc[fill] = sample.loc[fill] / sr.loc[fill]
    return sec


def sampling_rate(df):
    return safe_num(first_col(df, ['sampling_rate_hz', 'trace_sampling_rate_hz', 'sampling_rate']))


def trace_length(df):
    length = safe_num(first_col(df, ['trace_length_sec']))
    npts = safe_num(first_col(df, ['trace_npts', 'trace_n_samples', 'npts']))
    sr = sampling_rate(df)
    fill = ~np.isfinite(length) & np.isfinite(npts) & np.isfinite(sr) & (sr > 0)
    length = length.copy()
    length.loc[fill] = npts.loc[fill] / sr.loc[fill]
    return length


def distance_km(df):
    return safe_num(first_col(df, ['distance_km', 'source_distance_km', 'source_receiver_distance_km']))


def classify_rows(df):
    p = phase_seconds(df, 'p')
    s = phase_seconds(df, 's')
    sr = sampling_rate(df)
    dist = distance_km(df)
    raw_len = trace_length(df)
    reason = pd.Series('kept', index=df.index, dtype=object)

    missing_p = ~np.isfinite(p)
    reason.loc[missing_p] = 'missing_p'
    missing_sr = (reason == 'kept') & (~np.isfinite(sr) | (sr <= 0))
    reason.loc[missing_sr] = 'missing_sampling_rate'
    too_far = (reason == 'kept') & np.isfinite(dist) & (dist > MAX_DISTANCE_KM)
    reason.loc[too_far] = 'distance_gt_max'

    need_sp = (reason == 'kept') & ~np.isfinite(dist)
    missing_s = need_sp & ~np.isfinite(s)
    reason.loc[missing_s] = 'missing_distance_and_s'
    sp = s - p
    invalid_sp = need_sp & (reason == 'kept') & (~np.isfinite(sp) | (sp <= 0))
    reason.loc[invalid_sp] = 'invalid_sp_time'
    sp_too_large = need_sp & (reason == 'kept') & (sp > MAX_SP_SEC_WITHOUT_DISTANCE)
    reason.loc[sp_too_large] = 'sp_gt_max_without_distance'

    start = np.where(p >= P_TARGET_SEC, p - P_TARGET_SEC, EARLY_P_FALLBACK_START_SEC)
    start = np.maximum(start, 0.0)
    available_len = raw_len - start
    out = pd.DataFrame({
        'p_sec': p,
        's_sec': s,
        'sampling_rate_hz': sr,
        'distance_km': dist,
        'trace_length_sec': raw_len,
        'window_start_sec': start,
        'available_window_len_sec': available_len,
        'qc_reason': reason,
    })
    return out


def fill_missing_trace_lengths_from_waveforms(dataset, stats):
    mask = (stats['qc_reason'] == 'kept') & ~np.isfinite(stats['trace_length_sec'])
    if not bool(mask.any()):
        stats['trace_length_source'] = np.where(np.isfinite(stats['trace_length_sec']), 'metadata', '')
        return stats
    _, sbd = import_seisbench()
    _, cls = resolve_dataset_class(sbd, dataset)
    ds = construct_dataset(cls, DATA_ROOT / dataset / 'traces')
    idx_values = stats.loc[mask, 'sb_idx'].to_numpy(dtype=np.int64)
    sr_values = stats.loc[mask, 'sampling_rate_hz'].to_numpy(dtype=float)
    filled = np.full(len(idx_values), np.nan, dtype=float)
    for i, (sb_idx, sr) in enumerate(zip(idx_values, sr_values), start=1):
        if i % WAVEFORM_LENGTH_PROGRESS_EVERY == 0:
            print('{}: read waveform lengths {:,}/{:,}'.format(dataset, i, len(idx_values)), flush=True)
        try:
            if np.isfinite(sr) and sr > 0:
                filled[i - 1] = waveform_npts(get_waveform(ds, sb_idx)) / sr
        except Exception:
            filled[i - 1] = np.nan
    stats = stats.copy()
    stats['trace_length_source'] = np.where(np.isfinite(stats['trace_length_sec']), 'metadata', '')
    stats.loc[mask, 'trace_length_sec'] = filled
    stats.loc[mask & np.isfinite(stats['trace_length_sec']), 'trace_length_source'] = 'waveform'
    stats['available_window_len_sec'] = stats['trace_length_sec'] - stats['window_start_sec']
    still_missing = (stats['qc_reason'] == 'kept') & ~np.isfinite(stats['trace_length_sec'])
    stats.loc[still_missing, 'qc_reason'] = 'missing_trace_length'
    return stats


def summarize(dataset, stats):
    kept = stats[stats['qc_reason'] == 'kept'].copy()
    row = {
        'dataset': dataset,
        'num_rows': int(len(stats)),
        'num_after_nonwindow_qc': int(len(kept)),
    }
    if len(stats):
        row['frac_after_nonwindow_qc'] = len(kept) / len(stats)
    else:
        row['frac_after_nonwindow_qc'] = np.nan
    vals = kept['available_window_len_sec'].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    for key, func in [('min', np.min), ('p05', lambda x: np.percentile(x, 5)), ('p10', lambda x: np.percentile(x, 10)), ('p25', lambda x: np.percentile(x, 25)), ('median', np.median), ('p75', lambda x: np.percentile(x, 75)), ('p90', lambda x: np.percentile(x, 90)), ('p95', lambda x: np.percentile(x, 95)), ('max', np.max)]:
        row['available_len_{}'.format(key)] = float(func(vals)) if len(vals) else np.nan
    for threshold in THRESHOLDS_SEC:
        row['num_available_ge_{:g}s'.format(threshold)] = int((vals >= threshold).sum())
        row['frac_available_ge_{:g}s'.format(threshold)] = float((vals >= threshold).mean()) if len(vals) else np.nan
    reasons = stats['qc_reason'].value_counts()
    for reason, count in reasons.items():
        row['qc_{}'.format(reason)] = int(count)
    return row


def plot_combined(values):
    kept = values[values['qc_reason'] == 'kept']
    n = len(DATASETS)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASETS):
        vals = kept.loc[kept['dataset'] == dataset, 'available_window_len_sec'].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=BINS, color='#4C78A8', alpha=0.85, edgecolor='white', linewidth=0.25)
        for threshold in THRESHOLDS_SEC:
            ax.axvline(threshold, color='#D62728' if threshold == 50 else '#888888', linewidth=0.9, linestyle='--')
        ax.set_ylabel(dataset)
        ax.grid(True, axis='y', alpha=0.25)
        ax.text(0.99, 0.78, 'N={:,}'.format(len(vals)), ha='right', va='center', transform=ax.transAxes, fontsize=8)
    axes[-1].set_xlabel('Available window length after P-based start (s)')
    fig.supylabel('Trace count')
    fig.tight_layout()
    fig.savefig(FIG_COMBINED, dpi=DPI)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_values = []
    summary_rows = []
    for dataset in DATASETS:
        path = DATA_ROOT / dataset / 'phase_index.csv'
        if not path.exists():
            print('skip {}: missing {}'.format(dataset, path), flush=True)
            continue
        df = pd.read_csv(path, low_memory=False)
        stats = classify_rows(df)
        stats.insert(0, 'dataset', dataset)
        stats.insert(1, 'sb_idx', df['sb_idx'].to_numpy() if 'sb_idx' in df.columns else np.arange(len(df)))
        stats = fill_missing_trace_lengths_from_waveforms(dataset, stats)
        all_values.append(stats)
        summary_rows.append(summarize(dataset, stats))
        kept = stats[stats['qc_reason'] == 'kept']
        print('{}: rows {:,} | after non-window QC {:,}'.format(dataset, len(stats), len(kept)), flush=True)
    if not all_values:
        raise RuntimeError('No phase_index.csv files were found under {}'.format(DATA_ROOT))
    values = pd.concat(all_values, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    values.to_csv(VALUES_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    plot_combined(values)
    print('values: {}'.format(VALUES_CSV), flush=True)
    print('summary: {}'.format(SUMMARY_CSV), flush=True)
    print('figure: {}'.format(FIG_COMBINED), flush=True)
    show_cols = ['dataset', 'num_rows', 'num_after_nonwindow_qc', 'available_len_p10', 'available_len_median']
    show_cols += ['frac_available_ge_{:g}s'.format(item) for item in THRESHOLDS_SEC]
    print(summary[[col for col in show_cols if col in summary.columns]], flush=True)


if __name__ == '__main__':
    main()