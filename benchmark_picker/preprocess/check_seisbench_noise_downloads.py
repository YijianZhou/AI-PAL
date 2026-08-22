"""Check SeisBench noise download/index completeness.

Run after download_seisbench_noise_raw.py.  The checker opens the exact noise
source class/path recorded in noise/manifest.json, validates row counts, and
probes a few selected noise waveforms by sb_idx.
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

import check_seisbench_downloads as base


DATA_ROOT = base.DATA_ROOT
DATASETS = base.DATASETS
REPORT_CSV = DATA_ROOT / 'seisbench_noise_download_check.csv'
NOISE_SUBDIR = 'noise'
NOISE_INDEX_NAME = 'noise_index.csv'
NOISE_METADATA_NAME = 'metadata_noise_full.csv'
NOISE_MANIFEST_NAME = 'manifest.json'
PROBE_PER_DATASET = 3


def read_manifest(path):
    if not path.exists():
        return {}
    with open(path) as fp:
        return json.load(fp)


def resolve_class(sbd, class_name, dataset_name):
    if class_name:
        cls = getattr(sbd, class_name, None)
        if cls is not None:
            return class_name, cls
    return base.resolve_dataset_class(sbd, dataset_name)


def construct_dataset(cls, path, source_kwargs=None):
    kwargs = {
        'path': path,
        'metadata_cache': False,
        'missing_components': 'pad',
        'component_order': 'ENZ',
    }
    kwargs.update(source_kwargs or {})
    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop('missing_components', None)
        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop('component_order', None)
            return cls(**kwargs)


def read_probe_indices(noise_index_path, total_rows):
    if total_rows <= 0 or not noise_index_path.exists():
        return []
    positions = sorted(set([0, total_rows // 2, total_rows - 1]))[:PROBE_PER_DATASET]
    try:
        header = pd.read_csv(noise_index_path, nrows=0)
        if 'sb_idx' not in header.columns:
            return positions
        values = []
        need = set(positions)
        offset = 0
        for chunk in pd.read_csv(noise_index_path, usecols=['sb_idx'], chunksize=200000):
            for pos in sorted(list(need)):
                local = pos - offset
                if 0 <= local < len(chunk):
                    values.append(int(chunk.iloc[local]['sb_idx']))
                    need.remove(pos)
            if not need:
                break
            offset += len(chunk)
        return values
    except Exception:
        return positions


def noise_waveform_probe(ds, noise_index_path, total_rows):
    indices = read_probe_indices(noise_index_path, total_rows)
    if not indices:
        return False, '', 'no noise rows to probe'
    shapes = []
    for sb_idx in indices:
        waveform = base.get_waveform(ds, sb_idx)
        arr = np.asarray(waveform)
        if arr.size == 0:
            return False, ';'.join(shapes), 'empty waveform at sb_idx {}'.format(sb_idx)
        shapes.append('{}:{}'.format(sb_idx, tuple(arr.shape)))
    return True, ';'.join(shapes), ''


def check_noise_dataset(dataset_name, sbd):
    root = DATA_ROOT / dataset_name
    noise_dir = root / NOISE_SUBDIR
    noise_index = noise_dir / NOISE_INDEX_NAME
    noise_metadata = noise_dir / NOISE_METADATA_NAME
    manifest_path = noise_dir / NOISE_MANIFEST_NAME
    manifest = read_manifest(manifest_path)
    dataset_path = Path(manifest.get('dataset_path') or manifest.get('traces_dir') or (root / 'traces'))
    manifest_class = manifest.get('dataset_class', '')

    row = {
        'dataset': dataset_name,
        'status': 'unknown',
        'dataset_class': manifest_class,
        'dataset_path': str(dataset_path),
        'noise_dir': str(noise_dir),
        'source_kind': manifest.get('source_kind', ''),
        'selection_method': manifest.get('selection_method', ''),
        'source_kwargs': json.dumps(manifest.get('source_kwargs', {}), sort_keys=True),
        'dataset_metadata_rows': np.nan,
        'noise_index_rows': base.count_csv_rows(noise_index),
        'metadata_noise_full_rows': base.count_csv_rows(noise_metadata),
        'manifest_num_dataset_rows': manifest.get('num_dataset_rows', np.nan),
        'manifest_num_noise_rows': manifest.get('num_noise_rows', np.nan),
        'noise_index_exists': noise_index.exists(),
        'metadata_noise_full_exists': noise_metadata.exists(),
        'manifest_exists': manifest_path.exists(),
        'num_partial_files': 0,
        'partial_files_preview': '',
        'noise_waveform_probe_ok': False,
        'noise_waveform_probe_shapes': '',
        'row_count_detail': '',
        'error': '',
    }

    partial_files = base.find_partial_files(
        dataset_path,
        root / 'noise',
        base.CACHE_ROOT / 'seisbench' / dataset_name,
        base.TMP_ROOT,
    )
    row['num_partial_files'] = len(partial_files)
    row['partial_files_preview'] = ';'.join(str(path) for path in partial_files[:10])

    try:
        class_name, cls = resolve_class(sbd, manifest_class, dataset_name)
        row['dataset_class'] = class_name
        ds = construct_dataset(cls, dataset_path, manifest.get('source_kwargs', {}))
        row['dataset_metadata_rows'] = int(len(ds.metadata))
        if row['noise_index_exists'] and np.isfinite(row['noise_index_rows']) and int(row['noise_index_rows']) > 0:
            probe_ok, probe_shapes, probe_error = noise_waveform_probe(ds, noise_index, int(row['noise_index_rows']))
            row['noise_waveform_probe_ok'] = bool(probe_ok)
            row['noise_waveform_probe_shapes'] = probe_shapes
            if probe_error:
                row['error'] = probe_error
    except Exception as exc:
        row['status'] = 'error_opening_dataset'
        row['error'] = repr(exc)
        return row

    required_outputs = row['noise_index_exists'] and row['metadata_noise_full_exists'] and row['manifest_exists']
    counts_match = True
    details = []
    noise_rows = row['noise_index_rows']
    metadata_rows = row['metadata_noise_full_rows']
    manifest_noise_rows = row['manifest_num_noise_rows']
    manifest_dataset_rows = row['manifest_num_dataset_rows']
    dataset_rows = row['dataset_metadata_rows']

    if not np.isfinite(noise_rows):
        counts_match = False
        details.append('noise_index_rows missing')
    if not np.isfinite(metadata_rows) or (np.isfinite(noise_rows) and int(metadata_rows) != int(noise_rows)):
        counts_match = False
        details.append('metadata_noise_full_rows={} expected={}'.format(metadata_rows, noise_rows))
    if np.isfinite(manifest_noise_rows) and np.isfinite(noise_rows) and int(manifest_noise_rows) != int(noise_rows):
        counts_match = False
        details.append('manifest_num_noise_rows={} expected={}'.format(manifest_noise_rows, noise_rows))
    if np.isfinite(manifest_dataset_rows) and int(manifest_dataset_rows) != int(dataset_rows):
        counts_match = False
        details.append('manifest_num_dataset_rows={} expected={}'.format(manifest_dataset_rows, dataset_rows))
    row['row_count_detail'] = '; '.join(details)

    if row['num_partial_files'] > 0:
        row['status'] = 'has_partial_files_check_manually'
    elif not required_outputs:
        row['status'] = 'missing_noise_outputs'
    elif np.isfinite(noise_rows) and int(noise_rows) == 0:
        row['status'] = 'no_noise_rows'
    elif not counts_match:
        row['status'] = 'row_count_mismatch'
    elif not row['noise_waveform_probe_ok']:
        row['status'] = 'error_reading_noise_waveforms'
    else:
        row['status'] = 'ok'
    return row


def main():
    _, sbd = base.import_seisbench()
    rows = []
    for dataset in DATASETS:
        print('=' * 80)
        print('checking noise {}'.format(dataset), flush=True)
        try:
            row = check_noise_dataset(dataset, sbd)
        except Exception as exc:
            traceback.print_exc()
            row = {'dataset': dataset, 'status': 'error', 'error': repr(exc)}
        rows.append(row)
        print('{} noise: {}'.format(dataset, row.get('status')))
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
    cols = [
        'dataset', 'status', 'dataset_class', 'source_kind', 'selection_method', 'source_kwargs',
        'dataset_metadata_rows', 'noise_index_rows', 'metadata_noise_full_rows',
        'manifest_num_noise_rows', 'noise_waveform_probe_ok', 'num_partial_files',
        'row_count_detail', 'error'
    ]
    print(report[[col for col in cols if col in report.columns]])


if __name__ == '__main__':
    main()