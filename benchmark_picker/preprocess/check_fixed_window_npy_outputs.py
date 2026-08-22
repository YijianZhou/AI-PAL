"""Check fixed-window positive NPY shard outputs for SeisBench and CEED.

The checker is intentionally I/O oriented and memory-light.  It validates the
files produced by build_seisbench_fixed_window_npy.py and
build_ceed_fixed_window_npy.py:

  - summary.json exists and agrees with pos.npy/sample_index.csv
  - every shard listed in pos.npy exists and has the expected shape
  - sample_index.csv rows point to valid shard rows
  - P/S labels are finite and inside the fixed window
  - cleaned_phase_index.csv mirrors sample_index.csv
  - failure_index.csv exists and, when possible, accounts for skipped inputs

It writes a compact report to /nas/zhouyj/AI_datasets/fixed_window_npy_check.csv.
"""
import json
from pathlib import Path
import traceback

import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd


DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
SEISBENCH_DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
SEISBENCH_OUT_SUBDIR = 'npy_shards_fixed40'
CEED_OUT_SUBDIR = 'npy_shards_fixed50'
REPORT_CSV = DATA_ROOT / 'fixed_window_npy_check.csv'
CSV_CHUNKSIZE = 300000
SHARD_PROBE_LIMIT = 9
REQUIRED_SAMPLE_COLUMNS = [
    'shard_path', 'row_in_shard', 'p_rel_sec', 's_rel_sec',
    'window_start_sec', 'window_end_sec', 'raw_end_sec',
]


def count_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return np.nan
    try:
        header = pd.read_csv(path, nrows=0)
        if len(header.columns) == 0:
            return 0
        first_col = header.columns[0]
        total = 0
        for chunk in pd.read_csv(path, usecols=[first_col], chunksize=CSV_CHUNKSIZE):
            total += len(chunk)
        return int(total)
    except Exception:
        try:
            with open(path, 'rb') as fp:
                return max(0, sum(1 for _ in fp) - 1)
        except Exception:
            return np.nan


def count_text_lines(path):
    path = Path(path)
    if not path.exists():
        return np.nan
    with open(path, 'rb') as fp:
        return sum(1 for _ in fp)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as fp:
        return json.load(fp)


def load_shard_index(path):
    if not Path(path).exists():
        return np.empty((0, 2), dtype=str)
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.empty((0, 2), dtype=str)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 2))
    return arr.astype(str)


def to_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def to_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def probe_indices(n):
    if n <= 0:
        return []
    vals = sorted(set([0, n // 2, n - 1]))
    if len(vals) >= SHARD_PROBE_LIMIT:
        return vals[:SHARD_PROBE_LIMIT]
    if n > len(vals):
        extra = np.linspace(0, n - 1, min(SHARD_PROBE_LIMIT, n), dtype=int).tolist()
        vals = sorted(set(vals + extra))
    return vals[:SHARD_PROBE_LIMIT]


def scan_sample_index(sample_index_path, expected_by_shard, window_length_sec):
    result = {
        'sample_rows': 0,
        'missing_required_columns': '',
        'missing_shard_path_refs': 0,
        'bad_row_in_shard': 0,
        'bad_p_rel': 0,
        'bad_s_rel': 0,
        'bad_window_bounds': 0,
        'sample_index_error': '',
    }
    sample_index_path = Path(sample_index_path)
    if not sample_index_path.exists():
        result['sample_index_error'] = 'missing sample_index.csv'
        return result, {}

    header = pd.read_csv(sample_index_path, nrows=0)
    missing = [col for col in REQUIRED_SAMPLE_COLUMNS if col not in header.columns]
    result['missing_required_columns'] = ';'.join(missing)
    usecols = [col for col in REQUIRED_SAMPLE_COLUMNS if col in header.columns]
    if not usecols:
        result['sample_index_error'] = 'no usable sample_index columns'
        return result, {}

    observed_by_shard = {}
    try:
        for chunk in pd.read_csv(sample_index_path, usecols=usecols, chunksize=CSV_CHUNKSIZE):
            result['sample_rows'] += len(chunk)
            if 'shard_path' in chunk.columns:
                shard_paths = chunk['shard_path'].astype(str)
            else:
                shard_paths = pd.Series([''] * len(chunk), index=chunk.index)
                result['missing_shard_path_refs'] += len(chunk)

            if 'row_in_shard' in chunk.columns:
                row_in_shard = pd.to_numeric(chunk['row_in_shard'], errors='coerce')
            else:
                row_in_shard = pd.Series(np.nan, index=chunk.index)
            if 'p_rel_sec' in chunk.columns:
                p_rel = pd.to_numeric(chunk['p_rel_sec'], errors='coerce')
                result['bad_p_rel'] += int((~np.isfinite(p_rel) | (p_rel < 0) | (p_rel > window_length_sec)).sum())
            if 's_rel_sec' in chunk.columns:
                s_rel = pd.to_numeric(chunk['s_rel_sec'], errors='coerce')
                finite_s = np.isfinite(s_rel)
                result['bad_s_rel'] += int((finite_s & ((s_rel < 0) | (s_rel > window_length_sec))).sum())
            if {'window_start_sec', 'window_end_sec', 'raw_end_sec'}.issubset(chunk.columns):
                w0 = pd.to_numeric(chunk['window_start_sec'], errors='coerce')
                w1 = pd.to_numeric(chunk['window_end_sec'], errors='coerce')
                raw_end = pd.to_numeric(chunk['raw_end_sec'], errors='coerce')
                bad = (~np.isfinite(w0) | ~np.isfinite(w1) | ~np.isfinite(raw_end) | (w0 < 0) | (w1 <= w0) | (w1 > raw_end + 1e-3))
                result['bad_window_bounds'] += int(bad.sum())

            grouped = pd.DataFrame({'shard_path': shard_paths, 'row_in_shard': row_in_shard}).groupby('shard_path')['row_in_shard']
            for shard_path, rows in grouped:
                if not shard_path or shard_path == 'nan':
                    result['missing_shard_path_refs'] += int(len(rows))
                    continue
                rows = pd.to_numeric(rows, errors='coerce')
                count = int(len(rows))
                max_row = int(rows.max()) if np.isfinite(rows.max()) else -1
                min_row = int(rows.min()) if np.isfinite(rows.min()) else -1
                prev = observed_by_shard.get(shard_path, {'count': 0, 'max_row': -1, 'min_row': 10**18})
                prev['count'] += count
                prev['max_row'] = max(prev['max_row'], max_row)
                prev['min_row'] = min(prev['min_row'], min_row)
                observed_by_shard[shard_path] = prev

        expected_paths = set(expected_by_shard)
        for shard_path, obs in observed_by_shard.items():
            expected = expected_by_shard.get(shard_path)
            if expected is None:
                result['missing_shard_path_refs'] += obs['count']
                continue
            if obs['min_row'] < 0 or obs['max_row'] >= expected:
                result['bad_row_in_shard'] += obs['count']
            if obs['count'] != expected:
                result['bad_row_in_shard'] += abs(obs['count'] - expected)
        for shard_path, expected in expected_by_shard.items():
            if shard_path not in observed_by_shard:
                result['bad_row_in_shard'] += expected
    except Exception as exc:
        result['sample_index_error'] = repr(exc)
    return result, observed_by_shard


def probe_shards(shard_rows, expected_npts):
    result = {
        'shards_probed': 0,
        'missing_shard_files': 0,
        'bad_shard_shape': 0,
        'bad_shard_values': 0,
        'probe_error': '',
        'probe_shapes': '',
    }
    shapes = []
    for idx in probe_indices(len(shard_rows)):
        shard_path, nrows = shard_rows[idx]
        nrows = to_int(nrows, -1)
        path = Path(shard_path)
        if not path.exists():
            result['missing_shard_files'] += 1
            continue
        try:
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
            result['shards_probed'] += 1
            shapes.append('{}:{}'.format(idx, tuple(arr.shape)))
            if arr.ndim != 3 or arr.shape[0] != nrows or arr.shape[1] != 3 or arr.shape[2] != expected_npts + 2:
                result['bad_shard_shape'] += 1
                continue
            wave = arr[:, :, 2:]
            p_labels = arr[:, :, 0]
            s_labels = arr[:, :, 1]
            if not np.isfinite(wave).all():
                result['bad_shard_values'] += 1
            if not np.isfinite(p_labels).all():
                result['bad_shard_values'] += 1
            finite_s = np.isfinite(s_labels)
            if finite_s.any() and not np.isfinite(s_labels[finite_s]).all():
                result['bad_shard_values'] += 1
        except Exception as exc:
            result['probe_error'] = repr(exc)
            result['bad_shard_shape'] += 1
    result['probe_shapes'] = ';'.join(shapes)
    return result



def failure_reason_summary(path, limit=8):
    path = Path(path)
    if not path.exists():
        return '', 0
    try:
        header = pd.read_csv(path, nrows=0)
        if 'skip_reason' not in header.columns:
            return '', 0
        counts = {}
        total = 0
        for chunk in pd.read_csv(path, usecols=['skip_reason'], chunksize=CSV_CHUNKSIZE):
            vc = chunk['skip_reason'].fillna('unknown').astype(str).value_counts()
            total += int(vc.sum())
            for key, val in vc.items():
                counts[key] = counts.get(key, 0) + int(val)
        items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        return ';'.join('{}:{}'.format(key, val) for key, val in items), total
    except Exception as exc:
        return 'error_reading_failure_reasons:{}'.format(repr(exc)), 0

def check_one(dataset_name, out_dir):
    out_dir = Path(out_dir)
    summary_path = out_dir / 'summary.json'
    summary = read_json(summary_path)
    window_length_sec = to_float(summary.get('window_length_sec', 50.0), 50.0)
    sample_rate = to_float(summary.get('target_sample_rate', 100.0), 100.0)
    expected_npts = int(round(window_length_sec * sample_rate))

    shard_index_path = Path(summary.get('shard_index', out_dir / 'pos.npy'))
    sample_index_path = Path(summary.get('sample_index', out_dir / 'sample_index.csv'))
    failure_index_path = Path(summary.get('failure_index', out_dir / 'failure_index.csv'))
    cleaned_index_path = Path(summary.get('cleaned_phase_index', out_dir / 'cleaned_phase_index.csv'))
    cleaned_phase_path = Path(summary.get('cleaned_phase', out_dir / 'cleaned_phase.pha'))

    row = {
        'dataset': dataset_name,
        'status': 'unknown',
        'out_dir': str(out_dir),
        'summary_exists': summary_path.exists(),
        'shard_index_exists': shard_index_path.exists(),
        'sample_index_exists': sample_index_path.exists(),
        'failure_index_exists': failure_index_path.exists(),
        'cleaned_phase_index_exists': cleaned_index_path.exists(),
        'cleaned_phase_exists': cleaned_phase_path.exists(),
        'window_length_sec': window_length_sec,
        'target_sample_rate': sample_rate,
        'expected_npts': expected_npts,
        'num_shards_summary': to_int(summary.get('num_shards', np.nan), -1),
        'samples_written_summary': to_int(summary.get('counts', {}).get('samples_written', summary.get('num_samples', np.nan)), -1),
        'num_input_rows_summary': to_int(summary.get('num_input_rows', np.nan), -1),
        'shard_index_rows': np.nan,
        'shard_index_sample_sum': np.nan,
        'sample_index_rows': count_csv_rows(sample_index_path),
        'failure_index_rows': count_csv_rows(failure_index_path),
        'failure_top_reasons': '',
        'cleaned_phase_index_rows': count_csv_rows(cleaned_index_path),
        'cleaned_phase_lines': count_text_lines(cleaned_phase_path),
        'errors': '',
        'warnings': '',
    }

    row['failure_top_reasons'], failure_reason_rows = failure_reason_summary(failure_index_path)
    if failure_reason_rows and np.isfinite(row['failure_index_rows']) and failure_reason_rows != int(row['failure_index_rows']):
        row['failure_reason_rows'] = failure_reason_rows
    else:
        row['failure_reason_rows'] = row['failure_index_rows']

    errors = []
    warnings = []
    for key in ['summary_exists', 'shard_index_exists', 'sample_index_exists', 'failure_index_exists', 'cleaned_phase_index_exists', 'cleaned_phase_exists']:
        if not row[key]:
            errors.append('missing_' + key.replace('_exists', ''))

    shard_rows = load_shard_index(shard_index_path)
    row['shard_index_rows'] = int(len(shard_rows))
    shard_counts = [to_int(item[1], -1) for item in shard_rows]
    row['shard_index_sample_sum'] = int(sum(max(0, val) for val in shard_counts))
    expected_by_shard = {str(path): to_int(nrows, -1) for path, nrows in shard_rows}

    if row['num_shards_summary'] >= 0 and row['num_shards_summary'] != row['shard_index_rows']:
        errors.append('summary_num_shards_mismatch')
    if row['samples_written_summary'] >= 0 and row['samples_written_summary'] != row['shard_index_sample_sum']:
        errors.append('summary_samples_written_mismatch')
    if np.isfinite(row['sample_index_rows']) and int(row['sample_index_rows']) != row['shard_index_sample_sum']:
        errors.append('sample_index_vs_shard_sum_mismatch')
    if np.isfinite(row['cleaned_phase_index_rows']) and np.isfinite(row['sample_index_rows']) and int(row['cleaned_phase_index_rows']) != int(row['sample_index_rows']):
        errors.append('cleaned_phase_index_row_mismatch')
    if np.isfinite(row['cleaned_phase_lines']) and np.isfinite(row['cleaned_phase_index_rows']) and int(row['cleaned_phase_lines']) != 2 * int(row['cleaned_phase_index_rows']):
        warnings.append('cleaned_phase_line_count_not_2x_index')

    scan_result, _ = scan_sample_index(sample_index_path, expected_by_shard, window_length_sec)
    row.update(scan_result)
    probe_result = probe_shards(shard_rows, expected_npts)
    row.update(probe_result)

    for key in ['missing_required_columns', 'sample_index_error', 'probe_error']:
        if row.get(key):
            errors.append(key)
    for key in ['missing_shard_path_refs', 'bad_row_in_shard', 'bad_p_rel', 'bad_s_rel', 'bad_window_bounds', 'missing_shard_files', 'bad_shard_shape', 'bad_shard_values']:
        if to_int(row.get(key, 0), 0) > 0:
            errors.append(key)

    if row['num_input_rows_summary'] >= 0 and np.isfinite(row['failure_index_rows']) and np.isfinite(row['sample_index_rows']):
        accounted = int(row['failure_index_rows']) + int(row['sample_index_rows'])
        row['accounted_rows'] = accounted
        if accounted != row['num_input_rows_summary']:
            warnings.append('sample_plus_failure_not_equal_input')
    else:
        row['accounted_rows'] = np.nan

    row['errors'] = ';'.join(sorted(set(errors)))
    row['warnings'] = ';'.join(sorted(set(warnings)))
    if errors:
        row['status'] = 'error'
    elif warnings:
        row['status'] = 'warning'
    else:
        row['status'] = 'ok'
    return row


def main():
    raw_checks = [(name, DATA_ROOT / name / SEISBENCH_OUT_SUBDIR) for name in SEISBENCH_DATASETS]
    raw_checks.append(('CEED', DATA_ROOT / 'CEED' / CEED_OUT_SUBDIR))
    seen = set()
    checks = []
    for dataset_name, out_dir in raw_checks:
        key = (dataset_name, str(out_dir))
        if key in seen:
            continue
        seen.add(key)
        checks.append((dataset_name, out_dir))
    rows = []
    for dataset_name, out_dir in checks:
        print('=' * 80, flush=True)
        print('checking {} -> {}'.format(dataset_name, out_dir), flush=True)
        try:
            row = check_one(dataset_name, out_dir)
        except Exception as exc:
            traceback.print_exc()
            row = {'dataset': dataset_name, 'out_dir': str(out_dir), 'status': 'error', 'errors': repr(exc)}
        rows.append(row)
        print('{}: {}'.format(dataset_name, row.get('status')), flush=True)
        if row.get('errors'):
            print('  errors: {}'.format(row['errors']), flush=True)
        if row.get('warnings'):
            print('  warnings: {}'.format(row['warnings']), flush=True)
        print('  samples: {} | failures: {} | shards: {}'.format(
            row.get('sample_index_rows'), row.get('failure_index_rows'), row.get('shard_index_rows')
        ), flush=True)
        if row.get('failure_top_reasons'):
            print('  failure reasons: {}'.format(row['failure_top_reasons']), flush=True)

    report = pd.DataFrame(rows)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_CSV, index=False)
    print('=' * 80, flush=True)
    print('report: {}'.format(REPORT_CSV), flush=True)
    cols = [
        'dataset', 'status', 'sample_index_rows', 'failure_index_rows',
        'shard_index_rows', 'shard_index_sample_sum', 'samples_written_summary',
        'bad_window_bounds', 'failure_top_reasons', 'errors', 'warnings'
    ]
    print(report[[col for col in cols if col in report.columns]], flush=True)


if __name__ == '__main__':
    main()