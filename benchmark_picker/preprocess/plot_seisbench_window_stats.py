"""Plot SeisBench raw-window length and P-arrival buffer distributions.

The script reads /nas/zhouyj/AI_datasets/<dataset>/phase_index.csv.  If the
normalized index does not yet contain trace_npts or trace_length_sec, it falls
back to /nas/zhouyj/AI_datasets/<dataset>/metadata_full.csv and joins by sb_idx
(row order), avoiding any waveform glob/read step.
"""
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# I/O paths and plotting controls
# -----------------------------------------------------------------------------
DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
OUT_DIR = DATA_ROOT / 'figures_window_stats'
DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
DPI = 600
SAVE_FORMAT = 'jpg'

# Use finite percentile limits for clearer figures while keeping full stats in CSV.
WINDOW_LENGTH_XMAX_PERCENTILE = 99.5
P_BUFFER_XMAX_PERCENTILE = 99.5
MIN_VALID_SECONDS = 0.0
HIST_BINS = 120

NPTS_CANDIDATES = [
    'trace_npts',
    'trace_n_samples',
    'trace_sample_count',
    'trace_number_of_samples',
    'trace_length_samples',
    'trace_samples',
    'trace_num_samples',
    'trace_samples_count',
    'npts',
    'n_samples',
    'num_samples',
]

START_TIME_CANDIDATES = [
    'trace_start_time',
    'trace_starttime',
    'start_time',
    'starttime',
]

END_TIME_CANDIDATES = [
    'trace_end_time',
    'trace_endtime',
    'end_time',
    'endtime',
]

SR_CANDIDATES = [
    'sampling_rate_hz',
    'trace_sampling_rate_hz',
    'sampling_rate',
]

P_SAMPLE_CANDIDATES = [
    'p_arrival_sample',
    'trace_p_arrival_sample',
    'trace_P_arrival_sample',
    'trace_Pg_arrival_sample',
    'trace_Pn_arrival_sample',
]


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
def read_header(path):
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)


def first_existing(columns, candidates):
    columns = set(columns)
    for col in candidates:
        if col in columns:
            return col
    return None


def read_existing_columns(path, requested):
    header = read_header(path)
    usecols = [col for col in requested if col in header]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols)


def has_finite_numeric(df, column):
    if column not in df.columns:
        return False
    values = pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return bool(values.notna().any())


def add_trace_length_from_metadata(dataset_dir, df):
    if has_finite_numeric(df, 'trace_length_sec'):
        return df
    if has_finite_numeric(df, 'trace_npts') and has_finite_numeric(df, 'sampling_rate_hz'):
        return df

    metadata_path = dataset_dir / 'metadata_full.csv'
    if not metadata_path.exists():
        return df

    header = read_header(metadata_path)
    npts_col = first_existing(header, NPTS_CANDIDATES)
    start_col = first_existing(header, START_TIME_CANDIDATES)
    end_col = first_existing(header, END_TIME_CANDIDATES)
    sr_col = first_existing(header, SR_CANDIDATES)
    wanted = []
    for col in [npts_col, start_col, end_col]:
        if col is not None:
            wanted.append(col)
    if sr_col is not None and 'sampling_rate_hz' not in df.columns:
        wanted.append(sr_col)
    if not wanted:
        length_like = [
            col for col in header
            if any(key in col.lower() for key in ['npts', 'sample', 'length', 'duration', 'start', 'end'])
        ]
        print('  {}: no metadata length columns found; candidates seen: {}'.format(
            dataset_dir.name, ', '.join(length_like[:20])
        ))
        return df

    meta = pd.read_csv(metadata_path, usecols=list(dict.fromkeys(wanted)))
    meta['sb_idx'] = np.arange(len(meta), dtype=np.int64)
    if npts_col is not None:
        meta = meta.rename(columns={npts_col: 'trace_npts'})
    if sr_col is not None and sr_col != 'sampling_rate_hz':
        meta = meta.rename(columns={sr_col: 'sampling_rate_hz'})
    if start_col is not None and end_col is not None:
        start = pd.to_datetime(meta[start_col], errors='coerce', format='mixed')
        end = pd.to_datetime(meta[end_col], errors='coerce', format='mixed')
        meta['trace_length_sec'] = (end - start).dt.total_seconds()

    cols = ['sb_idx']
    if 'trace_length_sec' in meta.columns:
        cols.append('trace_length_sec')
    if 'trace_npts' in meta.columns:
        cols.append('trace_npts')
    if 'sampling_rate_hz' in meta.columns and 'sampling_rate_hz' not in df.columns:
        cols.append('sampling_rate_hz')
    return df.merge(meta[cols], on='sb_idx', how='left')



def numeric_column(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors='coerce')

def load_dataset_stats(dataset):
    dataset_dir = DATA_ROOT / dataset
    index_path = dataset_dir / 'phase_index.csv'
    if not index_path.exists():
        print('skip {}: missing {}'.format(dataset, index_path))
        return pd.DataFrame()

    header = read_header(index_path)
    npts_col = first_existing(header, NPTS_CANDIDATES)
    sr_col = first_existing(header, SR_CANDIDATES)
    p_sample_col = first_existing(header, P_SAMPLE_CANDIDATES)

    requested = ['sb_idx', 'trace_length_sec', 'p_arrival_sec']
    for col in [npts_col, sr_col, p_sample_col]:
        if col is not None:
            requested.append(col)
    # keep order stable and unique
    requested = list(dict.fromkeys(requested))
    df = read_existing_columns(index_path, requested)
    if df.empty:
        print('skip {}: no usable columns in {}'.format(dataset, index_path))
        return pd.DataFrame()
    if 'sb_idx' not in df.columns:
        df['sb_idx'] = np.arange(len(df), dtype=np.int64)

    rename = {}
    if npts_col is not None and npts_col != 'trace_npts':
        rename[npts_col] = 'trace_npts'
    if sr_col is not None and sr_col != 'sampling_rate_hz':
        rename[sr_col] = 'sampling_rate_hz'
    if p_sample_col is not None and p_sample_col != 'p_arrival_sample':
        rename[p_sample_col] = 'p_arrival_sample'
    df = df.rename(columns=rename)

    if 'trace_length_sec' in df.columns and not has_finite_numeric(df, 'trace_length_sec'):
        df = df.drop(columns=['trace_length_sec'])
    if 'trace_npts' in df.columns and not has_finite_numeric(df, 'trace_npts'):
        df = df.drop(columns=['trace_npts'])
    df = add_trace_length_from_metadata(dataset_dir, df)

    sr = numeric_column(df, 'sampling_rate_hz')
    if 'trace_length_sec' in df.columns:
        trace_length_sec = numeric_column(df, 'trace_length_sec')
    elif 'trace_npts' in df.columns:
        trace_length_sec = numeric_column(df, 'trace_npts') / sr
    else:
        trace_length_sec = pd.Series(np.nan, index=df.index)

    if 'p_arrival_sec' in df.columns:
        p_buffer_sec = numeric_column(df, 'p_arrival_sec')
    elif 'p_arrival_sample' in df.columns:
        p_buffer_sec = numeric_column(df, 'p_arrival_sample') / sr
    else:
        p_buffer_sec = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        'dataset': dataset,
        'sb_idx': df['sb_idx'].astype(np.int64),
        'trace_length_sec': trace_length_sec,
        'p_buffer_sec': p_buffer_sec,
    })
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[(out['trace_length_sec'] > MIN_VALID_SECONDS) | (out['p_buffer_sec'] > MIN_VALID_SECONDS)]
    print('{}: {:,} rows | length finite {:,} | p-buffer finite {:,}'.format(
        dataset,
        len(out),
        int(out['trace_length_sec'].notna().sum()),
        int(out['p_buffer_sec'].notna().sum()),
    ))
    return out


# -----------------------------------------------------------------------------
# Stats and plotting
# -----------------------------------------------------------------------------
def summarize_one(values):
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return {
            'count': 0,
            'min': np.nan,
            'p01': np.nan,
            'p05': np.nan,
            'p25': np.nan,
            'median': np.nan,
            'p75': np.nan,
            'p95': np.nan,
            'p99': np.nan,
            'max': np.nan,
        }
    return {
        'count': int(len(values)),
        'min': float(values.min()),
        'p01': float(values.quantile(0.01)),
        'p05': float(values.quantile(0.05)),
        'p25': float(values.quantile(0.25)),
        'median': float(values.quantile(0.50)),
        'p75': float(values.quantile(0.75)),
        'p95': float(values.quantile(0.95)),
        'p99': float(values.quantile(0.99)),
        'max': float(values.max()),
    }


def write_summary(all_stats):
    rows = []
    for dataset, group in all_stats.groupby('dataset', sort=False):
        for metric in ['trace_length_sec', 'p_buffer_sec']:
            row = {'dataset': dataset, 'metric': metric}
            row.update(summarize_one(group[metric]))
            rows.append(row)
    for metric in ['trace_length_sec', 'p_buffer_sec']:
        row = {'dataset': 'ALL', 'metric': metric}
        row.update(summarize_one(all_stats[metric]))
        rows.append(row)
    summary = pd.DataFrame(rows)
    path = OUT_DIR / 'seisbench_window_stats_summary.csv'
    summary.to_csv(path, index=False)
    print('summary: {}'.format(path))
    return summary


def finite_positive(values):
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return values[values >= MIN_VALID_SECONDS]


def robust_xlim(values, percentile):
    values = finite_positive(values)
    if len(values) == 0:
        return 1.0
    xmax = float(np.nanpercentile(values, percentile))
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = float(values.max())
    return max(xmax, 1.0)


def plot_combined(all_stats):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))

    metrics = [
        ('trace_length_sec', 'Raw window length (s)', WINDOW_LENGTH_XMAX_PERCENTILE),
        ('p_buffer_sec', 'P arrival time from window start (s)', P_BUFFER_XMAX_PERCENTILE),
    ]
    for ax, (metric, xlabel, pct) in zip(axes, metrics):
        xmax = robust_xlim(all_stats[metric], pct)
        bins = np.linspace(0, xmax, HIST_BINS + 1)
        for color, dataset in zip(colors, DATASETS):
            vals = finite_positive(all_stats.loc[all_stats['dataset'] == dataset, metric])
            if len(vals) == 0:
                continue
            ax.hist(vals.clip(upper=xmax), bins=bins, histtype='step', density=True,
                    linewidth=1.3, color=color, label='{} ({:,})'.format(dataset, len(vals)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_xlim(0, xmax)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[1].legend(fontsize=7, loc='upper right', frameon=False)
    path = OUT_DIR / ('seisbench_window_stats_combined.%s' % SAVE_FORMAT)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print('figure: {}'.format(path))


def plot_per_dataset(all_stats):
    for dataset in DATASETS:
        group = all_stats[all_stats['dataset'] == dataset]
        if group.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
        specs = [
            ('trace_length_sec', 'Raw window length (s)', WINDOW_LENGTH_XMAX_PERCENTILE),
            ('p_buffer_sec', 'P arrival time from window start (s)', P_BUFFER_XMAX_PERCENTILE),
        ]
        for ax, (metric, xlabel, pct) in zip(axes, specs):
            vals = finite_positive(group[metric])
            if len(vals) == 0:
                ax.text(0.5, 0.5, 'No finite values', ha='center', va='center')
                ax.set_axis_off()
                continue
            xmax = robust_xlim(vals, pct)
            bins = np.linspace(0, xmax, HIST_BINS + 1)
            ax.hist(vals.clip(upper=xmax), bins=bins, color='#4477aa', alpha=0.82)
            stats = summarize_one(vals)
            ax.axvline(stats['median'], color='#cc6677', linewidth=1.2, label='median')
            ax.axvline(stats['p95'], color='#228833', linewidth=1.1, linestyle='--', label='p95')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_xlim(0, xmax)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle('{} window statistics'.format(dataset), fontsize=12)
        path = OUT_DIR / ('{}_window_stats.{}'.format(dataset, SAVE_FORMAT))
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print('figure: {}'.format(path))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in DATASETS:
        df = load_dataset_stats(dataset)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError('No dataset stats loaded from {}'.format(DATA_ROOT))

    all_stats = pd.concat(frames, ignore_index=True)
    stats_path = OUT_DIR / 'seisbench_window_stats_values.csv'
    all_stats.to_csv(stats_path, index=False)
    print('values: {}'.format(stats_path))
    write_summary(all_stats)
    plot_combined(all_stats)
    plot_per_dataset(all_stats)


if __name__ == '__main__':
    main()