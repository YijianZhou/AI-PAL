"""Summarize fixed-window NPY conversion failures and suspicious rows.

This is a lightweight diagnostic companion to check_fixed_window_npy_outputs.py.
It prints success/failure rates, top skip_reason counts, and for any dataset
with bad_window_bounds reported by the checker, it writes example rows to help
judge whether the issue is numerical tolerance or real metadata inconsistency.
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


DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024', 'CEED']
SEISBENCH_OUT_SUBDIR = 'npy_shards_fixed40'
CEED_OUT_SUBDIR = 'npy_shards_fixed50'
CHECK_REPORT = DATA_ROOT / 'fixed_window_npy_check.csv'
SUMMARY_CSV = DATA_ROOT / 'fixed_window_failure_reason_summary.csv'
CWA_BAD_WINDOW_EXAMPLES = DATA_ROOT / 'CWA_bad_window_bound_examples.csv'
CHUNKSIZE = 300000
WINDOW_TOL_SEC = 1e-3


def count_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return 0
    header = pd.read_csv(path, nrows=0)
    if len(header.columns) == 0:
        return 0
    first_col = header.columns[0]
    total = 0
    for chunk in pd.read_csv(path, usecols=[first_col], chunksize=CHUNKSIZE):
        total += len(chunk)
    return int(total)


def reason_counts(path):
    path = Path(path)
    if not path.exists():
        return {}
    header = pd.read_csv(path, nrows=0)
    if 'skip_reason' not in header.columns:
        return {}
    counts = {}
    for chunk in pd.read_csv(path, usecols=['skip_reason'], chunksize=CHUNKSIZE):
        vc = chunk['skip_reason'].fillna('unknown').astype(str).value_counts()
        for key, val in vc.items():
            counts[key] = counts.get(key, 0) + int(val)
    return counts


def summarize_dataset(dataset):
    out_dir = DATA_ROOT / dataset / (CEED_OUT_SUBDIR if dataset == 'CEED' else SEISBENCH_OUT_SUBDIR)
    sample_index = out_dir / 'sample_index.csv'
    failure_index = out_dir / 'failure_index.csv'
    n_success = count_csv_rows(sample_index)
    n_failure = count_csv_rows(failure_index)
    total = n_success + n_failure
    counts = reason_counts(failure_index)
    rows = []
    if counts:
        for reason, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            rows.append({
                'dataset': dataset,
                'skip_reason': reason,
                'count': int(count),
                'fraction_of_failures': count / n_failure if n_failure else np.nan,
                'fraction_of_input': count / total if total else np.nan,
                'num_success': n_success,
                'num_failure': n_failure,
                'num_input_accounted': total,
                'success_fraction': n_success / total if total else np.nan,
            })
    else:
        rows.append({
            'dataset': dataset,
            'skip_reason': '',
            'count': 0,
            'fraction_of_failures': np.nan,
            'fraction_of_input': np.nan,
            'num_success': n_success,
            'num_failure': n_failure,
            'num_input_accounted': total,
            'success_fraction': n_success / total if total else np.nan,
        })
    return rows


def write_bad_window_examples(dataset='CWA', limit=200):
    out_dir = DATA_ROOT / dataset / (CEED_OUT_SUBDIR if dataset == 'CEED' else SEISBENCH_OUT_SUBDIR)
    sample_index = out_dir / 'sample_index.csv'
    if not sample_index.exists():
        return 0
    header = pd.read_csv(sample_index, nrows=0)
    cols = [
        'dataset', 'sb_idx', 'station_key', 'trace_name', 'p_rel_sec', 's_rel_sec',
        'window_start_sec', 'window_end_sec', 'raw_end_sec', 'trace_length_sec',
        'sampling_rate_hz', 'shard_path', 'row_in_shard'
    ]
    usecols = [col for col in cols if col in header.columns]
    if not {'window_start_sec', 'window_end_sec', 'raw_end_sec'}.issubset(usecols):
        return 0
    examples = []
    for chunk in pd.read_csv(sample_index, usecols=usecols, chunksize=CHUNKSIZE):
        w0 = pd.to_numeric(chunk['window_start_sec'], errors='coerce')
        w1 = pd.to_numeric(chunk['window_end_sec'], errors='coerce')
        raw_end = pd.to_numeric(chunk['raw_end_sec'], errors='coerce')
        bad = (~np.isfinite(w0) | ~np.isfinite(w1) | ~np.isfinite(raw_end) | (w0 < 0) | (w1 <= w0) | (w1 > raw_end + WINDOW_TOL_SEC))
        if bool(bad.any()):
            part = chunk.loc[bad].copy()
            part['window_end_minus_raw_end_sec'] = pd.to_numeric(part['window_end_sec'], errors='coerce') - pd.to_numeric(part['raw_end_sec'], errors='coerce')
            examples.append(part)
            if sum(len(item) for item in examples) >= limit:
                break
    if not examples:
        return 0
    out = pd.concat(examples, ignore_index=True).head(limit)
    out.to_csv(CWA_BAD_WINDOW_EXAMPLES, index=False)
    return len(out)


def main():
    rows = []
    for dataset in DATASETS:
        rows.extend(summarize_dataset(dataset))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print('summary: {}'.format(SUMMARY_CSV), flush=True)
    show = summary.sort_values(['dataset', 'count'], ascending=[True, False]).groupby('dataset').head(8)
    print(show[['dataset', 'skip_reason', 'count', 'fraction_of_failures', 'fraction_of_input', 'success_fraction']], flush=True)

    if CHECK_REPORT.exists():
        check = pd.read_csv(CHECK_REPORT)
        bad = check[(check.get('bad_window_bounds', 0).fillna(0).astype(int) > 0)] if 'bad_window_bounds' in check.columns else pd.DataFrame()
        if not bad.empty:
            for dataset in bad['dataset'].astype(str).unique():
                if dataset == 'CWA':
                    n = write_bad_window_examples(dataset)
                    if n:
                        print('bad window examples: {} ({:,} rows)'.format(CWA_BAD_WINDOW_EXAMPLES, n), flush=True)


if __name__ == '__main__':
    main()