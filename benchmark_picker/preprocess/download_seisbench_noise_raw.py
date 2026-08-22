"""Index/download SeisBench noise traces for positive-picker stability tests.

Several benchmark datasets expose noise through dedicated SeisBench dataset
classes rather than as rows inside the event dataset metadata.  This script
prefers those dedicated classes when available, writes a compact noise_index.csv
under /nas/zhouyj/AI_datasets/<dataset>/noise, and records the exact source
class/path in the manifest so later preprocessing reopens the correct dataset.
"""
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

OUT_ROOT = Path('/nas/zhouyj/AI_datasets')
CACHE_ROOT = OUT_ROOT / '_cache'
TMP_ROOT = OUT_ROOT / '_tmp'
SUMMARY_CSV = OUT_ROOT / 'seisbench_noise_download_summary.csv'
FORCE_DOWNLOAD = False
WAIT_FOR_FILE = True


def configure_large_file_cache():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
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


configure_large_file_cache()

import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd

import download_seisbench_raw as raw

COMMON_KWARGS = raw.COMMON_KWARGS

# Source candidates are tried in order.  ``all_rows`` is used only for classes
# that are already noise-only.  Metadata-filtered sources keep conservative
# explicit/fallback selection from the full event-style metadata.
NOISE_DATASETS = [
    {
        'name': 'INSTANCE',
        'sources': [
            {'class_aliases': ['InstanceNoise'], 'path_suffix': 'noise/traces', 'selection': 'all_rows'},
            {'class_aliases': ['InstanceCountsCombined'], 'path_suffix': 'noise/traces_combined', 'selection': 'metadata'},
            {'class_aliases': ['InstanceCounts'], 'path_suffix': 'traces', 'selection': 'metadata'},
        ],
    },
    {
        'name': 'CWA',
        'sources': [
            {'class_aliases': ['CWANoise'], 'path_suffix': 'noise/traces', 'selection': 'all_rows'},
            {'class_aliases': ['CWA'], 'path_suffix': 'noise/traces', 'selection': 'all_rows', 'kwargs': {'chunks': ['_noise1', '_noise2']}},
            {'class_aliases': ['CWA'], 'path_suffix': 'traces', 'selection': 'metadata'},
        ],
    },
    {
        'name': 'PNW',
        'sources': [
            {'class_aliases': ['PNWNoise'], 'path_suffix': 'noise/traces', 'selection': 'all_rows'},
            {'class_aliases': ['PNW'], 'path_suffix': 'traces', 'selection': 'metadata'},
        ],
    },
    {
        'name': 'STEAD',
        'sources': [
            {'class_aliases': ['STEAD'], 'path_suffix': 'traces', 'selection': 'metadata'},
        ],
    },
    {
        'name': 'piSDL',
        'sources': [
            {'class_aliases': ['piSDL', 'PiSDL', 'PISDL', 'pISDL'], 'path_suffix': 'traces', 'selection': 'metadata'},
        ],
    },
    {
        'name': 'OBST2024',
        'sources': [
            {'class_aliases': ['OBST2024'], 'path_suffix': 'traces', 'selection': 'metadata'},
        ],
    },
]

NOISE_LABEL_COLUMNS = [
    'trace_category', 'trace_type', 'trace_label', 'trace_class', 'trace_source',
    'source_type', 'source_category', 'event_type', 'trace_name', 'trace_id', 'id',
]
EVENT_ID_COLUMNS = ['source_id', 'event_id', 'source_event_id']


def import_seisbench():
    import seisbench as sb
    import seisbench.data as sbd
    try:
        sb.cache_root = CACHE_ROOT / 'seisbench'
    except Exception:
        pass
    return sb, sbd


def resolve_dataset_class(sbd, aliases):
    for alias in aliases:
        cls = getattr(sbd, alias, None)
        if cls is not None:
            return alias, cls
    raise AttributeError('No SeisBench class found for aliases {}'.format(aliases))


def construct_dataset(cls, path, kwargs):
    call_kwargs = dict(COMMON_KWARGS)
    call_kwargs.update(kwargs or {})
    call_kwargs.update({
        'path': path,
        'force': FORCE_DOWNLOAD,
        'wait_for_file': WAIT_FOR_FILE,
    })
    try:
        return cls(**call_kwargs)
    except TypeError:
        call_kwargs.pop('force', None)
        call_kwargs.pop('wait_for_file', None)
        try:
            return cls(**call_kwargs)
        except TypeError:
            call_kwargs.pop('component_order', None)
            return cls(**call_kwargs)


def has_noise_text(series):
    text = series.fillna('').astype(str).str.lower()
    return text.str.contains('noise', regex=False) | text.isin(['n', 'noisy', 'ambient_noise', 'noise_trace'])


def explicit_noise_mask(full_df):
    mask = pd.Series(False, index=full_df.index)
    used_cols = []
    for col in NOISE_LABEL_COLUMNS:
        if col not in full_df.columns:
            continue
        col_mask = has_noise_text(full_df[col])
        if bool(col_mask.any()):
            used_cols.append(col)
            mask |= col_mask
    return mask, used_cols


def fallback_noise_mask(full_df, index_df):
    p = pd.to_numeric(index_df.get('p_arrival_sample', np.nan), errors='coerce')
    s = pd.to_numeric(index_df.get('s_arrival_sample', np.nan), errors='coerce')
    no_picks = p.isna() & s.isna()
    no_event = pd.Series(True, index=index_df.index)
    for col in EVENT_ID_COLUMNS:
        if col in full_df.columns:
            text = full_df[col].fillna('').astype(str).str.strip().str.lower()
            no_event &= text.isin(['', 'nan', 'none', '-1'])
    if 'source_lat' in index_df.columns:
        no_event &= pd.to_numeric(index_df['source_lat'], errors='coerce').isna()
    return no_picks & no_event


def choose_noise_rows(selection, full_df, index_df):
    if selection == 'all_rows':
        return pd.Series(True, index=index_df.index), 'all_rows_from_noise_dataset'
    explicit_mask, used_cols = explicit_noise_mask(full_df)
    if bool(explicit_mask.any()):
        return explicit_mask, 'explicit_noise_label:' + ','.join(used_cols)
    return fallback_noise_mask(full_df, index_df), 'fallback_no_picks_no_event'


def normalize_noise_index(index_df, noise_mask, source_kind):
    out = index_df.loc[noise_mask].copy().reset_index(drop=True)
    if out.empty:
        return out
    out['is_noise'] = True
    out['noise_source_kind'] = source_kind
    out['p_arrival_sample'] = np.nan
    out['s_arrival_sample'] = np.nan
    out['p_arrival_sec'] = np.nan
    out['s_arrival_sec'] = np.nan
    out['p_arrival_time'] = ''
    out['s_arrival_time'] = ''
    out['event_id'] = ''
    out['origin_time'] = ''
    out['source_lat'] = np.nan
    out['source_lon'] = np.nan
    out['source_depth_km'] = np.nan
    out['source_mag'] = np.nan
    return out


def try_source(dataset_name, source, sbd):
    dataset_root = OUT_ROOT / dataset_name
    traces_dir = dataset_root / source['path_suffix']
    traces_dir.mkdir(parents=True, exist_ok=True)
    class_name, cls = resolve_dataset_class(sbd, source['class_aliases'])
    ds = construct_dataset(cls, traces_dir, source.get('kwargs', {}))
    index_df, full_df = raw.normalize_metadata(dataset_name, class_name, traces_dir, ds)
    noise_mask, method = choose_noise_rows(source['selection'], full_df, index_df)
    source_kind = 'noise_dataset' if source['selection'] == 'all_rows' else 'metadata_filter'
    noise_index = normalize_noise_index(index_df, noise_mask, source_kind)
    noise_full = full_df.loc[noise_mask].copy().reset_index(drop=True)
    return {
        'class_name': class_name,
        'dataset_path': traces_dir,
        'dataset_obj': ds,
        'index_df': index_df,
        'full_df': full_df,
        'noise_index': noise_index,
        'noise_full': noise_full,
        'selection_method': method,
        'source_kind': source_kind,
        'source_spec': source,
    }


def process_dataset(spec, sbd):
    t0 = time.time()
    dataset_name = spec['name']
    dataset_root = OUT_ROOT / dataset_name
    noise_dir = dataset_root / 'noise'
    dataset_root.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80, flush=True)
    print('indexing noise {} -> {}'.format(dataset_name, noise_dir), flush=True)
    errors = []
    result = None
    for source in spec['sources']:
        try:
            print('  trying {} at {}'.format(source['class_aliases'], dataset_root / source['path_suffix']), flush=True)
            candidate = try_source(dataset_name, source, sbd)
            print('    source rows: {:,} | noise rows: {:,} | method: {}'.format(
                len(candidate['index_df']), len(candidate['noise_index']), candidate['selection_method']
            ), flush=True)
            if result is None or len(candidate['noise_index']) > len(result['noise_index']):
                result = candidate
            if len(candidate['noise_index']) > 0:
                break
        except Exception as exc:
            errors.append('{}: {}'.format(source['class_aliases'], repr(exc)))
            print('    failed: {}'.format(repr(exc)), flush=True)

    if result is None:
        raise RuntimeError('All noise sources failed for {}: {}'.format(dataset_name, ' | '.join(errors)))

    noise_index_path = noise_dir / 'noise_index.csv'
    noise_full_path = noise_dir / 'metadata_noise_full.csv'
    manifest_path = noise_dir / 'manifest.json'
    result['noise_index'].to_csv(noise_index_path, index=False)
    result['noise_full'].to_csv(noise_full_path, index=False)

    manifest = {
        'dataset': dataset_name,
        'dataset_class': result['class_name'],
        'dataset_path': str(result['dataset_path']),
        'traces_dir': str(result['dataset_path']),
        'noise_dir': str(noise_dir),
        'noise_index_csv': str(noise_index_path),
        'metadata_noise_full_csv': str(noise_full_path),
        'num_dataset_rows': int(len(result['index_df'])),
        'num_noise_rows': int(len(result['noise_index'])),
        'selection_method': result['selection_method'],
        'source_kind': result['source_kind'],
        'source_aliases': result['source_spec']['class_aliases'],
        'source_kwargs': result['source_spec'].get('kwargs', {}),
        'attempt_errors': errors,
        'elapsed_sec': time.time() - t0,
    }
    with open(manifest_path, 'w') as fp:
        json.dump(manifest, fp, indent=2)

    print('  selected class: {}'.format(result['class_name']), flush=True)
    print('  selected path: {}'.format(result['dataset_path']), flush=True)
    print('  dataset rows: {:,}'.format(len(result['index_df'])), flush=True)
    print('  noise rows: {:,}'.format(len(result['noise_index'])), flush=True)
    print('  method: {}'.format(result['selection_method']), flush=True)
    print('  index: {}'.format(noise_index_path), flush=True)
    return {
        'dataset': dataset_name,
        'dataset_class': result['class_name'],
        'status': 'ok',
        'dataset_path': str(result['dataset_path']),
        'num_dataset_rows': int(len(result['index_df'])),
        'num_noise_rows': int(len(result['noise_index'])),
        'selection_method': result['selection_method'],
        'source_kind': result['source_kind'],
        'noise_index_csv': str(noise_index_path),
        'elapsed_sec': manifest['elapsed_sec'],
        'error': '',
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sb, sbd = import_seisbench()
    print('output root: {}'.format(OUT_ROOT), flush=True)
    print('seisbench cache_root: {}'.format(getattr(sb, 'cache_root', 'unknown')), flush=True)
    rows = []
    for spec in NOISE_DATASETS:
        try:
            rows.append(process_dataset(spec, sbd))
        except Exception as exc:
            traceback.print_exc()
            rows.append({
                'dataset': spec['name'],
                'dataset_class': '',
                'status': 'error',
                'dataset_path': '',
                'num_dataset_rows': 0,
                'num_noise_rows': 0,
                'selection_method': '',
                'source_kind': '',
                'noise_index_csv': '',
                'elapsed_sec': np.nan,
                'error': repr(exc),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print('=' * 80, flush=True)
    print('summary: {}'.format(SUMMARY_CSV), flush=True)
    print(summary[['dataset', 'dataset_class', 'status', 'num_noise_rows', 'selection_method', 'source_kind', 'error']], flush=True)


if __name__ == '__main__':
    main()