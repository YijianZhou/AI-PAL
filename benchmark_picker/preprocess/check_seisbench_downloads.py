"""Check SeisBench benchmark download/index completeness.

Run after download_seisbench_raw.py.  The strongest check is instantiating each
SeisBench dataset from its local traces directory and comparing that metadata
count against phase_index.csv and manifest.json.  The script also reports likely
partial/temp files that may have been left by interrupted downloads.
"""
import json
import os
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


DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
CACHE_ROOT = DATA_ROOT / '_cache'
TMP_ROOT = DATA_ROOT / '_tmp'
DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
REPORT_CSV = DATA_ROOT / 'seisbench_download_check.csv'

DATASET_CLASS_ALIASES = {
    'INSTANCE': ['InstanceCounts'],
    'CWA': ['CWA'],
    'PNW': ['PNW'],
    'STEAD': ['STEAD'],
    'piSDL': ['piSDL', 'PiSDL', 'PISDL', 'pISDL'],
    'OBST2024': ['OBST2024'],
}

PARTIAL_SUFFIXES = (
    '.part', '.partial', '.tmp', '.download', '.incomplete', '.crdownload',
)
PARTIAL_NAME_TOKENS = (
    '.part', 'partial', 'tmp', 'download', 'incomplete', 'lock',
)


def configure_cache_env():
    env_paths = {
        'SEISBENCH_CACHE_ROOT': CACHE_ROOT / 'seisbench',
        'XDG_CACHE_HOME': CACHE_ROOT / 'xdg',
        'POOCH_HOME': CACHE_ROOT / 'pooch',
        'MPLCONFIGDIR': CACHE_ROOT / 'matplotlib',
        'NUMBA_CACHE_DIR': CACHE_ROOT / 'numba',
        'TMPDIR': TMP_ROOT,
        'TEMP': TMP_ROOT,
        'TMP': TMP_ROOT,
    }
    for key, value in env_paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    tempfile.tempdir = str(TMP_ROOT)


def install_pandas_mixed_datetime_fallback():
    original_to_datetime = pd.to_datetime

    def to_datetime_with_mixed_fallback(*args, **kwargs):
        try:
            return original_to_datetime(*args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if (
                "doesn't match format" not in message
                and "does not match format" not in message
                and "time data" not in message
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
        sb.cache_root = CACHE_ROOT / 'seisbench'
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


def count_csv_rows(path):
    if not path.exists():
        return np.nan
    try:
        header = pd.read_csv(path, nrows=0)
        if len(header.columns) == 0:
            return 0
        first_col = header.columns[0]
        total = 0
        for chunk in pd.read_csv(path, usecols=[first_col], chunksize=200000):
            total += len(chunk)
        return int(total)
    except Exception:
        return int(sum(1 for _ in open(path, 'rb')) - 1)


def count_hdf5_files(path):
    if not path.exists():
        return 0
    return len(list(path.rglob('*.hdf5'))) + len(list(path.rglob('*.h5')))


def count_all_files(path):
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob('*') if item.is_file())


def dir_size_gb(path):
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total / 1024**3


def find_partial_files(*roots):
    hits = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            name = path.name.lower()
            if path.suffix.lower() in PARTIAL_SUFFIXES or any(token in name for token in PARTIAL_NAME_TOKENS):
                hits.append(path)
    return sorted(hits)



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


def waveform_probe(ds, count):
    if count <= 0:
        return False, '', ''
    indices = sorted(set([0, count // 2, count - 1]))
    shapes = []
    for index in indices:
        waveform = get_waveform(ds, index)
        arr = np.asarray(waveform)
        if arr.size == 0:
            return False, ';'.join(shapes), 'empty waveform at index {}'.format(index)
        shapes.append('{}:{}'.format(index, tuple(arr.shape)))
    return True, ';'.join(shapes), ''

def read_manifest(path):
    if not path.exists():
        return {}
    with open(path) as fp:
        return json.load(fp)


def check_dataset(dataset_name, sbd):
    root = DATA_ROOT / dataset_name
    traces_dir = root / 'traces'
    manifest_path = root / 'manifest.json'
    metadata_full = root / 'metadata_full.csv'
    phase_index = root / 'phase_index.csv'
    phase_file = root / 'phase.pha'

    row = {
        'dataset': dataset_name,
        'status': 'unknown',
        'traces_dir': str(traces_dir),
        'class_name': '',
        'dataset_metadata_rows': np.nan,
        'metadata_full_rows': count_csv_rows(metadata_full),
        'phase_index_rows': count_csv_rows(phase_index),
        'manifest_num_traces': np.nan,
        'num_hdf5_files': count_hdf5_files(traces_dir),
        'num_files_in_traces': count_all_files(traces_dir),
        'traces_size_gb': dir_size_gb(traces_dir),
        'waveform_probe_ok': False,
        'waveform_probe_shapes': '',
        'metadata_full_exists': metadata_full.exists(),
        'phase_index_exists': phase_index.exists(),
        'phase_file_exists': phase_file.exists(),
        'manifest_exists': manifest_path.exists(),
        'num_partial_files': 0,
        'partial_files_preview': '',
        'row_count_detail': '',
        'error': '',
    }

    manifest = read_manifest(manifest_path)
    if manifest:
        row['manifest_num_traces'] = manifest.get('num_traces', np.nan)

    partial_files = find_partial_files(traces_dir, CACHE_ROOT / 'seisbench' / dataset_name, TMP_ROOT)
    row['num_partial_files'] = len(partial_files)
    row['partial_files_preview'] = ';'.join(str(path) for path in partial_files[:10])

    try:
        class_name, cls = resolve_dataset_class(sbd, dataset_name)
        row['class_name'] = class_name
        ds = construct_dataset(cls, traces_dir)
        row['dataset_metadata_rows'] = int(len(ds.metadata))
        probe_ok, probe_shapes, probe_error = waveform_probe(ds, row['dataset_metadata_rows'])
        row['waveform_probe_ok'] = bool(probe_ok)
        row['waveform_probe_shapes'] = probe_shapes
        if probe_error:
            row['error'] = probe_error
    except Exception as exc:
        row['status'] = 'error_opening_dataset'
        row['error'] = repr(exc)
        return row

    expected = row['dataset_metadata_rows']
    counts_match = True
    details = []
    for key in ('metadata_full_rows', 'phase_index_rows'):
        value = row[key]
        if not np.isfinite(value) or int(value) != int(expected):
            counts_match = False
            details.append('{}={} expected={}'.format(key, value, expected))
    if np.isfinite(row['manifest_num_traces']) and int(row['manifest_num_traces']) != int(expected):
        counts_match = False
        details.append('manifest_num_traces={} expected={}'.format(row['manifest_num_traces'], expected))
    row['row_count_detail'] = '; '.join(details)

    required_outputs = row['metadata_full_exists'] and row['phase_index_exists'] and row['phase_file_exists'] and row['manifest_exists']
    if not row['waveform_probe_ok']:
        row['status'] = 'error_reading_waveforms'
    elif row['num_partial_files'] > 0:
        row['status'] = 'has_partial_files_check_manually'
    elif not required_outputs:
        row['status'] = 'missing_index_outputs'
    elif not counts_match:
        row['status'] = 'row_count_mismatch'
    else:
        row['status'] = 'ok'
    return row


def main():
    _, sbd = import_seisbench()
    rows = []
    for dataset in DATASETS:
        print('=' * 80)
        print('checking {}'.format(dataset), flush=True)
        try:
            row = check_dataset(dataset, sbd)
        except Exception as exc:
            traceback.print_exc()
            row = {'dataset': dataset, 'status': 'error', 'error': repr(exc)}
        rows.append(row)
        print('{}: {}'.format(dataset, row.get('status')))
        if row.get('row_count_detail'):
            print('  row counts: {}'.format(row['row_count_detail']))
        if row.get('error'):
            print('  error: {}'.format(row['error']))
        if row.get('num_partial_files', 0):
            print('  partial preview: {}'.format(row.get('partial_files_preview', '')))

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_CSV, index=False)
    print('=' * 80)
    print('report: {}'.format(REPORT_CSV))
    print(report[['dataset', 'status', 'dataset_metadata_rows', 'metadata_full_rows', 'phase_index_rows', 'manifest_num_traces', 'waveform_probe_ok', 'num_partial_files', 'row_count_detail', 'error']])


if __name__ == '__main__':
    main()