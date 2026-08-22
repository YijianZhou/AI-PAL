"""Download and index SeisBench benchmark datasets for picker evaluation.

This first-stage script keeps the datasets in native SeisBench format under
/nas/zhouyj/AI_datasets/<dataset>/traces.  It also writes compact metadata
indexes so the next preprocessing step can iterate by SeisBench row index
instead of scanning files with glob.
"""
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

# -----------------------------------------------------------------------------
# I/O paths and cache control
# -----------------------------------------------------------------------------
OUT_ROOT = Path('/nas/zhouyj/AI_datasets')
CACHE_ROOT = OUT_ROOT / '_cache'
TMP_ROOT = OUT_ROOT / '_tmp'
SUMMARY_CSV = OUT_ROOT / 'seisbench_download_summary.csv'
FORCE_DOWNLOAD = False
WAIT_FOR_FILE = True


def configure_large_file_cache():
    """Force downloader/cache/temp files away from the system disk."""
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


def install_pandas_mixed_datetime_fallback():
    """Retry Pandas datetime parsing for SeisBench metadata with mixed precision.

    Some SeisBench metadata files mix timestamps with and without fractional
    seconds.  Newer Pandas versions can infer one strict format and then raise
    on the other; format='mixed' is the intended fallback for this case.
    """
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


install_pandas_mixed_datetime_fallback()

# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------
# The first alias that exists in the local SeisBench version will be used.
DATASETS = [
    {
        'name': 'INSTANCE',
        'class_aliases': ['InstanceCounts'],
        'kwargs': {},
    },
    {
        'name': 'CWA',
        'class_aliases': ['CWA'],
        'kwargs': {},
    },
    {
        'name': 'PNW',
        'class_aliases': ['PNW'],
        'kwargs': {},
    },
    {
        'name': 'STEAD',
        'class_aliases': ['STEAD'],
        'kwargs': {},
    },
    {
        'name': 'piSDL',
        'class_aliases': ['piSDL', 'PiSDL', 'PISDL', 'pISDL'],
        'kwargs': {},
    },
    {
        'name': 'OBST2024',
        'class_aliases': ['OBST2024'],
        'kwargs': {},
    },
]

# Keep constructor kwargs conservative here. Component order/sampling-rate choices
# belong to the next preprocessing step, where we know how the picker should see
# each waveform. metadata_cache=False keeps startup memory low for large datasets.
COMMON_KWARGS = {
    'metadata_cache': False,
    'component_order': 'ENZ',
    'missing_components': 'pad',
}


PICK_SAMPLE_CANDIDATES = {
    'p': [
        'trace_p_arrival_sample',
        'trace_P_arrival_sample',
        'trace_Pg_arrival_sample',
        'trace_Pn_arrival_sample',
        'p_arrival_sample',
    ],
    's': [
        'trace_s_arrival_sample',
        'trace_S_arrival_sample',
        'trace_Sg_arrival_sample',
        'trace_Sn_arrival_sample',
        's_arrival_sample',
    ],
}

PICK_TIME_CANDIDATES = {
    'p': [
        'trace_p_arrival_time',
        'trace_P_arrival_time',
        'p_arrival_time',
    ],
    's': [
        'trace_s_arrival_time',
        'trace_S_arrival_time',
        's_arrival_time',
    ],
}

FIELD_CANDIDATES = {
    'trace_name': ['trace_name', 'trace_id', 'id'],
    'trace_chunk': ['trace_chunk', 'chunk'],
    'split': ['split', 'trace_split'],
    'sampling_rate_hz': ['trace_sampling_rate_hz', 'sampling_rate_hz', 'sampling_rate'],
    'trace_npts': ['trace_npts', 'trace_n_samples', 'trace_sample_count', 'trace_number_of_samples', 'trace_length_samples', 'trace_samples', 'npts'],
    'network': ['station_network_code', 'receiver_network_code', 'network_code'],
    'station': ['station_code', 'receiver_code', 'receiver_station_code'],
    'location': ['station_location_code', 'receiver_location_code', 'location_code'],
    'channel': ['trace_channel', 'station_channel_code', 'receiver_channel_code', 'channel_code'],
    'event_id': ['source_id', 'event_id', 'source_event_id'],
    'origin_time': ['source_origin_time', 'origin_time'],
    'source_lat': ['source_latitude_deg', 'source_latitude', 'event_latitude'],
    'source_lon': ['source_longitude_deg', 'source_longitude', 'event_longitude'],
    'source_depth_km': ['source_depth_km', 'source_depth', 'event_depth_km'],
    'source_mag': ['source_magnitude', 'source_magnitude_value', 'event_magnitude'],
    'receiver_lat': ['receiver_latitude', 'station_latitude_deg', 'station_latitude'],
    'receiver_lon': ['receiver_longitude', 'station_longitude_deg', 'station_longitude'],
    'receiver_elev_m': ['receiver_elevation_m', 'station_elevation_m'],
    'distance_km': [
        'source_distance_km',
        'source_receiver_distance_km',
        'path_ep_distance_km',
        'ep_distance_km',
        'distance_km',
    ],
}


# -----------------------------------------------------------------------------
# Small metadata helpers
# -----------------------------------------------------------------------------
def import_seisbench():
    import seisbench as sb
    import seisbench.data as sbd
    try:
        sb.cache_root = CACHE_ROOT / 'seisbench'
    except Exception:
        pass
    return sb, sbd


def first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def find_pick_column(columns, phase):
    exact = first_existing(columns, PICK_SAMPLE_CANDIDATES[phase])
    if exact is not None:
        return exact

    cols_lower = {col.lower(): col for col in columns}
    for lower, col in cols_lower.items():
        if 'arrival_sample' not in lower:
            continue
        tokens = lower.replace('-', '_').split('_')
        if phase in tokens or any(tok.startswith(phase) for tok in tokens):
            return col
    return None


def find_pick_time_column(columns, phase):
    exact = first_existing(columns, PICK_TIME_CANDIDATES[phase])
    if exact is not None:
        return exact

    cols_lower = {col.lower(): col for col in columns}
    for lower, col in cols_lower.items():
        if 'arrival_time' not in lower:
            continue
        tokens = lower.replace('-', '_').split('_')
        if phase in tokens or any(tok.startswith(phase) for tok in tokens):
            return col
    return None


def series_or_default(df, column, default=np.nan):
    if column is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[column]


def safe_string(value, default=''):
    if pd.isna(value):
        return default
    text = str(value)
    return text if text.lower() != 'nan' else default


def to_float_or_nan(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def make_station_key(row):
    net = safe_string(row.get('network'), '--')
    sta = safe_string(row.get('station'), '--')
    loc = safe_string(row.get('location'), '')
    chn = safe_string(row.get('channel'), '')
    return '.'.join([net, sta, loc, chn])


def list_dataset_files(dataset_path):
    dataset_path = Path(dataset_path)
    hdf5_files = sorted(
        str(path) for ext in ('*.hdf5', '*.h5')
        for path in dataset_path.rglob(ext)
    )
    metadata_files = sorted(str(path) for path in dataset_path.rglob('*.csv'))
    all_files = sorted(str(path) for path in dataset_path.rglob('*') if path.is_file())
    return hdf5_files, metadata_files, all_files


def resolve_dataset_class(sbd, aliases):
    for alias in aliases:
        cls = getattr(sbd, alias, None)
        if cls is not None:
            return alias, cls
    available = sorted(name for name in dir(sbd) if not name.startswith('_'))
    raise AttributeError(
        'None of the SeisBench dataset classes exist: {}. Available examples: {}'.format(
            aliases, ', '.join(available[:80])
        )
    )


def construct_dataset(cls, path, kwargs):
    call_kwargs = dict(COMMON_KWARGS)
    call_kwargs.update(kwargs)
    call_kwargs.update({
        'path': path,
        'force': FORCE_DOWNLOAD,
        'wait_for_file': WAIT_FOR_FILE,
    })
    try:
        return cls(**call_kwargs)
    except TypeError:
        # Some older/newer dataset constructors may not accept force/wait_for_file
        # explicitly. Keep the path and reader behavior kwargs, then retry.
        call_kwargs.pop('force', None)
        call_kwargs.pop('wait_for_file', None)
        try:
            return cls(**call_kwargs)
        except TypeError:
            call_kwargs.pop('component_order', None)
            return cls(**call_kwargs)


def normalize_metadata(dataset_name, class_name, dataset_path, ds):
    df = ds.metadata.copy()
    df = df.reset_index(drop=True)
    columns = set(df.columns)

    col = {key: first_existing(columns, vals) for key, vals in FIELD_CANDIDATES.items()}
    p_col = find_pick_column(columns, 'p')
    s_col = find_pick_column(columns, 's')
    p_time_col = find_pick_time_column(columns, 'p')
    s_time_col = find_pick_time_column(columns, 's')

    out = pd.DataFrame({
        'sb_idx': np.arange(len(df), dtype=np.int64),
        'dataset': dataset_name,
        'dataset_class': class_name,
        'dataset_path': str(dataset_path),
        'trace_name': series_or_default(df, col['trace_name'], ''),
        'trace_chunk': series_or_default(df, col['trace_chunk'], ''),
        'split': series_or_default(df, col['split'], ''),
        'sampling_rate_hz': series_or_default(df, col['sampling_rate_hz'], np.nan),
        'trace_npts': series_or_default(df, col['trace_npts'], np.nan),
        'network': series_or_default(df, col['network'], ''),
        'station': series_or_default(df, col['station'], ''),
        'location': series_or_default(df, col['location'], ''),
        'channel': series_or_default(df, col['channel'], ''),
        'event_id': series_or_default(df, col['event_id'], ''),
        'origin_time': series_or_default(df, col['origin_time'], ''),
        'source_lat': series_or_default(df, col['source_lat'], np.nan),
        'source_lon': series_or_default(df, col['source_lon'], np.nan),
        'source_depth_km': series_or_default(df, col['source_depth_km'], np.nan),
        'source_mag': series_or_default(df, col['source_mag'], np.nan),
        'receiver_lat': series_or_default(df, col['receiver_lat'], np.nan),
        'receiver_lon': series_or_default(df, col['receiver_lon'], np.nan),
        'receiver_elev_m': series_or_default(df, col['receiver_elev_m'], np.nan),
        'distance_km': series_or_default(df, col['distance_km'], np.nan),
        'p_arrival_sample': series_or_default(df, p_col, np.nan),
        's_arrival_sample': series_or_default(df, s_col, np.nan),
        'p_arrival_time': series_or_default(df, p_time_col, ''),
        's_arrival_time': series_or_default(df, s_time_col, ''),
    })

    # Derive seconds from samples when sample counts/rates are available.
    sr = pd.to_numeric(out['sampling_rate_hz'], errors='coerce')
    npts = pd.to_numeric(out['trace_npts'], errors='coerce')
    out['trace_length_sec'] = npts / sr

    # Derive relative pick seconds when samples and sampling rates are available.
    p_samp = pd.to_numeric(out['p_arrival_sample'], errors='coerce')
    s_samp = pd.to_numeric(out['s_arrival_sample'], errors='coerce')
    out['p_arrival_sec'] = p_samp / sr
    out['s_arrival_sec'] = s_samp / sr
    out['station_key'] = out.apply(make_station_key, axis=1)

    # Preserve the original column mapping for auditability.
    out.attrs['normalized_column_mapping'] = {
        **col,
        'p_arrival_sample': p_col,
        's_arrival_sample': s_col,
        'p_arrival_time': p_time_col,
        's_arrival_time': s_time_col,
    }
    return out, df


def write_phase_file(path, index_df):
    """Write a light, human-readable phase file for quick inspection.

    Pick times are relative seconds from trace start when sample/rate metadata is
    available.  The normalized CSV remains the authoritative machine-readable
    index for the next stage.
    """
    work = index_df.copy()
    event_values = work['event_id'].fillna('').astype(str)
    trace_values = work['trace_name'].fillna('').astype(str)
    fallback_event = work['dataset'].astype(str) + ':' + trace_values
    work['_event_key'] = np.where(event_values.str.len() > 0, event_values, fallback_event)

    with open(path, 'w') as fp:
        for event_id, group in work.groupby('_event_key', sort=False):
            first = group.iloc[0]
            origin = safe_string(first.get('origin_time'), 'NaT')
            lat = first.get('source_lat', np.nan)
            lon = first.get('source_lon', np.nan)
            dep = first.get('source_depth_km', np.nan)
            mag = first.get('source_mag', np.nan)
            fp.write('{},{},{},{},{},{}\n'.format(origin, lat, lon, dep, mag, event_id))
            for _, row in group.iterrows():
                sta = safe_string(row.get('station_key'), '--.--..')
                tp = to_float_or_nan(row.get('p_arrival_sec'))
                ts = to_float_or_nan(row.get('s_arrival_sec'))
                dist = to_float_or_nan(row.get('distance_km'))
                fp.write('{},{},{},{},{},{}\n'.format(
                    sta,
                    tp,
                    ts,
                    dist,
                    int(row['sb_idx']),
                    safe_string(row.get('trace_name')),
                ))


def process_dataset(spec, sbd):
    t0 = time.time()
    dataset_name = spec['name']
    dataset_root = OUT_ROOT / dataset_name
    traces_dir = dataset_root / 'traces'
    dataset_root.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    class_name, cls = resolve_dataset_class(sbd, spec['class_aliases'])
    print('=' * 80)
    print('downloading/indexing {} using seisbench.data.{} -> {}'.format(
        dataset_name, class_name, traces_dir
    ))

    ds = construct_dataset(cls, traces_dir, spec.get('kwargs', {}))
    num_traces = len(ds.metadata)
    hdf5_files, metadata_files, all_files = list_dataset_files(traces_dir)

    index_df, full_df = normalize_metadata(dataset_name, class_name, traces_dir, ds)
    full_metadata_path = dataset_root / 'metadata_full.csv'
    index_path = dataset_root / 'phase_index.csv'
    phase_path = dataset_root / 'phase.pha'
    manifest_path = dataset_root / 'manifest.json'

    full_df.to_csv(full_metadata_path, index=False)
    index_df.to_csv(index_path, index=False)
    write_phase_file(phase_path, index_df)

    mapping = index_df.attrs.get('normalized_column_mapping', {})
    manifest = {
        'dataset': dataset_name,
        'dataset_class': class_name,
        'dataset_root': str(dataset_root),
        'traces_dir': str(traces_dir),
        'metadata_full_csv': str(full_metadata_path),
        'phase_index_csv': str(index_path),
        'phase_file': str(phase_path),
        'num_traces': int(num_traces),
        'num_hdf5_files': len(hdf5_files),
        'num_files_in_traces': len(all_files),
        'num_metadata_files': len(metadata_files),
        'hdf5_files': hdf5_files,
        'metadata_files': metadata_files,
        'normalized_column_mapping': mapping,
        'seisbench_path_property': str(getattr(ds, 'path', traces_dir)),
        'seisbench_chunks': [str(chunk) for chunk in getattr(ds, 'chunks', [])],
        'elapsed_sec': time.time() - t0,
    }
    with open(manifest_path, 'w') as fp:
        json.dump(manifest, fp, indent=2)

    print('  traces: {:,}'.format(num_traces))
    print('  files in traces: {} (hdf5: {})'.format(len(all_files), len(hdf5_files)))
    print('  index: {}'.format(index_path))
    print('  phase: {}'.format(phase_path))
    print('  elapsed: {:.1f}s'.format(manifest['elapsed_sec']))

    return {
        'dataset': dataset_name,
        'dataset_class': class_name,
        'status': 'ok',
        'traces_dir': str(traces_dir),
        'num_traces': int(num_traces),
        'num_hdf5_files': len(hdf5_files),
        'num_files_in_traces': len(all_files),
        'metadata_full_csv': str(full_metadata_path),
        'phase_index_csv': str(index_path),
        'phase_file': str(phase_path),
        'manifest_json': str(manifest_path),
        'elapsed_sec': manifest['elapsed_sec'],
        'error': '',
    }


def print_cache_locations():
    print('output root: {}'.format(OUT_ROOT))
    print('cache root: {}'.format(CACHE_ROOT))
    print('tmp root: {}'.format(TMP_ROOT))
    for key in ('SEISBENCH_CACHE_ROOT', 'XDG_CACHE_HOME', 'POOCH_HOME', 'TMPDIR'):
        print('  {}={}'.format(key, os.environ.get(key, '')))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print_cache_locations()
    sb, sbd = import_seisbench()
    print('seisbench cache_root: {}'.format(getattr(sb, 'cache_root', 'unknown')))

    rows = []
    for spec in DATASETS:
        try:
            rows.append(process_dataset(spec, sbd))
        except Exception as exc:
            print('ERROR while processing {}: {}'.format(spec['name'], repr(exc)))
            traceback.print_exc()
            rows.append({
                'dataset': spec['name'],
                'dataset_class': '',
                'status': 'error',
                'traces_dir': str(OUT_ROOT / spec['name'] / 'traces'),
                'num_traces': 0,
                'num_hdf5_files': 0,
                'num_files_in_traces': 0,
                'metadata_full_csv': '',
                'phase_index_csv': '',
                'phase_file': '',
                'manifest_json': '',
                'elapsed_sec': np.nan,
                'error': repr(exc),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print('=' * 80)
    print('summary: {}'.format(SUMMARY_CSV))
    print(summary[['dataset', 'dataset_class', 'status', 'num_traces', 'num_files_in_traces', 'num_hdf5_files', 'error']])


if __name__ == '__main__':
    main()