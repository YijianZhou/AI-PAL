"""Build fixed-window NPY shards from SeisBench noise traces.

Noise shard format intentionally matches the positive event-window format:
    shard.shape = (n_samples, 3, win_npts + 2)
    shard[:, :, 0] = NaN  # no P reference
    shard[:, :, 1] = NaN  # no S reference
    shard[:, :, 2:] = preprocessed waveform, 3 channels, 100 Hz

The output is meant for picker stability/hallucination tests.
"""
from collections import Counter
import hashlib
import json
import multiprocessing as mp
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

import build_seisbench_fixed_window_npy as fixed


DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
DATASETS = ['INSTANCE', 'CWA', 'PNW', 'STEAD', 'piSDL', 'OBST2024']
NOISE_INDEX_SUBDIR = 'noise'
NOISE_INDEX_NAME = 'noise_index.csv'
OUT_SUBDIR = 'npy_noise_shards_fixed50'
SUMMARY_CSV = DATA_ROOT / 'seisbench_noise_fixed_window_npy_summary.csv'

TARGET_SAMPLE_RATE = fixed.TARGET_SAMPLE_RATE
WINDOW_LENGTH_SEC = 50.0
NOISE_WINDOW_START_MODE = 'random_uniform'
RANDOM_SEED = 20250708
SHARD_SIZE = 1024
TASK_CHUNK_SIZE = 1024
NUM_WORKERS = 8
PROGRESS_EVERY_TASKS = 100
CLEAN_OUTPUT = True

DATASET_SETTINGS = {
    'window_length_sec': WINDOW_LENGTH_SEC,
    'p_target_sec': 0.0,
    'noise_window_start_mode': NOISE_WINDOW_START_MODE,
    'random_seed': RANDOM_SEED,
}

_WORKER_DATASET = None
_WORKER_DATASET_NAME = None
_WORKER_OUT_DIR = None


def resolve_dataset_class_by_name(sbd, class_name, dataset_name):
    if class_name:
        cls = getattr(sbd, class_name, None)
        if cls is not None:
            return class_name, cls
    return fixed.resolve_dataset_class(sbd, dataset_name)


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


def init_worker(dataset_name, dataset_path, out_dir, class_name='', source_kwargs=None):
    global _WORKER_DATASET, _WORKER_DATASET_NAME, _WORKER_OUT_DIR
    fixed.install_pandas_mixed_datetime_fallback()
    _, sbd = fixed.import_seisbench()
    _, cls = resolve_dataset_class_by_name(sbd, class_name, dataset_name)
    _WORKER_DATASET = construct_dataset(cls, Path(dataset_path), source_kwargs)
    _WORKER_DATASET_NAME = dataset_name
    _WORKER_OUT_DIR = Path(out_dir)


def reset_output_dir(out_dir):
    fixed.reset_output_dir(out_dir)


def load_noise_index(dataset_name):
    path = DATA_ROOT / dataset_name / NOISE_INDEX_SUBDIR / NOISE_INDEX_NAME
    if not path.exists():
        print('skip {}: missing {}'.format(dataset_name, path), flush=True)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'sb_idx' not in df.columns:
        df['sb_idx'] = np.arange(len(df), dtype=np.int64)
    return df



def read_noise_manifest(dataset_name):
    path = DATA_ROOT / dataset_name / NOISE_INDEX_SUBDIR / 'manifest.json'
    if not path.exists():
        return {}
    with open(path) as fp:
        return json.load(fp)


def get_noise_source(dataset_name, df):
    manifest = read_noise_manifest(dataset_name)
    dataset_path = manifest.get('dataset_path') or manifest.get('traces_dir')
    class_name = manifest.get('dataset_class', '')
    if not dataset_path and 'dataset_path' in df.columns and len(df) > 0:
        dataset_path = fixed.safe_str(df.iloc[0].get('dataset_path'), '')
    if not class_name and 'dataset_class' in df.columns and len(df) > 0:
        class_name = fixed.safe_str(df.iloc[0].get('dataset_class'), '')
    if not dataset_path:
        dataset_path = str(DATA_ROOT / dataset_name / 'traces')
    return Path(dataset_path), class_name, manifest

def row_to_noise_info(row):
    sr = fixed.safe_float(fixed.get_first(row, ['sampling_rate_hz', 'trace_sampling_rate_hz', 'sampling_rate']))
    trace_len = fixed.safe_float(fixed.get_first(row, ['trace_length_sec']))
    if not np.isfinite(trace_len):
        npts = fixed.safe_float(fixed.get_first(row, ['trace_npts', 'trace_n_samples', 'npts']))
        if np.isfinite(npts) and np.isfinite(sr) and sr > 0:
            trace_len = npts / sr
    return {
        'sb_idx': int(row['sb_idx']),
        'dataset': fixed.safe_str(row.get('dataset'), _WORKER_DATASET_NAME or ''),
        'sampling_rate_hz': sr,
        'p_sec': np.nan,
        's_sec': np.nan,
        'distance_km': np.nan,
        'trace_length_sec': trace_len,
        'station_key': fixed.safe_str(row.get('station_key'), ''),
        'trace_name': fixed.safe_str(row.get('trace_name'), ''),
        'trace_chunk': fixed.safe_str(row.get('trace_chunk'), ''),
        'channel': fixed.safe_str(row.get('channel'), ''),
        'units': fixed.safe_str(fixed.get_first(row, ['trace_units', 'trace_unit', 'units'], default='')),
        'event_id': '',
        'origin_time': '',
        'source_lat': np.nan,
        'source_lon': np.nan,
        'source_depth_km': np.nan,
        'source_mag': np.nan,
        'receiver_lat': fixed.safe_float(row.get('receiver_lat', np.nan)),
        'receiver_lon': fixed.safe_float(row.get('receiver_lon', np.nan)),
        'is_noise': True,
    }


def deterministic_noise_start(info, raw_end_sec):
    max_start_sec = raw_end_sec - WINDOW_LENGTH_SEC
    if max_start_sec < 0:
        return np.nan
    if max_start_sec == 0:
        return 0.0
    key = '{}|{}|{}|{}'.format(
        info.get('dataset', ''), info.get('sb_idx', ''),
        info.get('trace_name', ''), info.get('trace_chunk', '')
    )
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    seed = (int(digest[:8], 16) + int(RANDOM_SEED)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.0, max_start_sec))


def cut_noise_window(data, info):
    win_npts = int(round(WINDOW_LENGTH_SEC * TARGET_SAMPLE_RATE))
    raw_end_sec = data.shape[1] / float(TARGET_SAMPLE_RATE)
    if data.shape[1] < win_npts:
        return None, fixed.cut_failure('raw_window_shorter_than_fixed_window', 0.0, WINDOW_LENGTH_SEC, raw_end_sec)

    start_sec = deterministic_noise_start(info, raw_end_sec)
    if not np.isfinite(start_sec):
        return None, fixed.cut_failure('fixed_window_failed', np.nan, np.nan, raw_end_sec)
    start_idx = int(round(start_sec * TARGET_SAMPLE_RATE))
    start_idx = max(0, min(start_idx, data.shape[1] - win_npts))
    end_idx = start_idx + win_npts
    start_sec = start_idx / float(TARGET_SAMPLE_RATE)
    end_sec = end_idx / float(TARGET_SAMPLE_RATE)

    if start_idx < 0:
        return None, fixed.cut_failure('fixed_window_start_before_raw_start', start_sec, end_sec, raw_end_sec)
    if end_idx > data.shape[1]:
        return None, fixed.cut_failure('fixed_window_end_after_raw_end', start_sec, end_sec, raw_end_sec)
    out = np.zeros((3, win_npts + 2), dtype=np.float32)
    out[:, 0] = np.nan
    out[:, 1] = np.nan
    out[:, 2:] = data[:, start_idx:end_idx]
    return out, {
        'window_start_sec': start_sec,
        'window_end_sec': end_sec,
        'raw_end_sec': raw_end_sec,
        'p_rel_sec': np.nan,
        's_rel_sec': np.nan,
        'is_noise': True,
        'noise_window_start_mode': NOISE_WINDOW_START_MODE,
        'noise_random_seed': RANDOM_SEED,
    }


def failure_meta(info, reason, extra=None):
    meta = dict(info)
    meta['dataset'] = _WORKER_DATASET_NAME
    meta['skip_reason'] = reason
    if extra:
        meta.update(extra)
    return meta


def process_one_record(row):
    counts = Counter()
    info = row_to_noise_info(row)
    if not np.isfinite(info['sampling_rate_hz']) or info['sampling_rate_hz'] <= 0:
        counts['skipped_missing_sampling_rate'] += 1
        return None, None, failure_meta(info, 'missing_sampling_rate'), counts
    try:
        waveform = fixed.get_waveform(_WORKER_DATASET, info['sb_idx'])
        data, prep_counts, prep_info = fixed.preprocess_waveform(waveform, info)
        counts.update(prep_counts)
        if data is None:
            counts['skipped_bad_waveform'] += 1
            return None, None, failure_meta(info, 'bad_waveform'), counts
        sample, cut_info = cut_noise_window(data, info)
        if sample is None:
            reason = cut_info.get('skip_reason', 'fixed_window_failed')
            counts['skipped_' + reason] += 1
            return None, None, failure_meta(info, reason, cut_info), counts
        sample = fixed.normalize_channels(sample)
        meta = dict(info)
        meta.update(prep_info)
        meta.update(cut_info)
        meta['dataset'] = _WORKER_DATASET_NAME
        counts['samples_written'] += 1
        return sample, meta, None, counts
    except Exception as exc:
        counts['skipped_exception'] += 1
        return None, None, failure_meta(info, 'exception', {'error': repr(exc)}), counts


def write_task_outputs(task_idx, samples, metas, failures):
    shard_rows = []
    if samples:
        shard_dir = _WORKER_OUT_DIR / 'shards'
        index_dir = _WORKER_OUT_DIR / 'index_parts'
        shard_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        shard = np.stack(samples, axis=0).astype(np.float32, copy=False)
        shard_path = shard_dir / ('%s_noise_shard_%06d.npy' % (_WORKER_DATASET_NAME, task_idx))
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
    return shard_rows


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
    shard_rows = write_task_outputs(task_idx, samples, metas, failures)
    counts['tasks_done'] += 1
    return counts, shard_rows


def iter_tasks(df, chunk_size):
    task_idx = 0
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        yield task_idx, chunk.to_dict(orient='records')
        task_idx += 1


def write_noise_metadata_file(path, sample_index_path):
    header = list(pd.read_csv(sample_index_path, nrows=0).columns)
    wanted = [
        'dataset', 'sb_idx', 'station_key', 'trace_name', 'trace_chunk', 'channel',
        'window_start_sec', 'window_end_sec', 'raw_end_sec', 'shard_path', 'row_in_shard', 'is_noise'
    ]
    usecols = [col for col in wanted if col in header]
    if not usecols:
        pd.DataFrame().to_csv(path, index=False)
        return path
    chunks = pd.read_csv(sample_index_path, usecols=usecols, chunksize=200000)
    first = True
    with open(path, 'w') as fout:
        for chunk in chunks:
            chunk.to_csv(fout, index=False, header=first)
            first = False
    return path


def process_dataset(dataset_name):
    df = load_noise_index(dataset_name)
    if df.empty:
        return {'dataset': dataset_name, 'status': 'missing_or_empty_noise_index'}
    dataset_path, class_name, source_manifest = get_noise_source(dataset_name, df)
    out_dir = DATA_ROOT / dataset_name / OUT_SUBDIR
    reset_output_dir(out_dir)
    print('=' * 80, flush=True)
    print('building noise {} -> {}'.format(dataset_name, out_dir), flush=True)
    print('noise rows: {:,}'.format(len(df)), flush=True)

    counts = Counter()
    shard_rows = []
    tasks = iter_tasks(df, TASK_CHUNK_SIZE)
    if NUM_WORKERS <= 1:
        init_worker(dataset_name, dataset_path, out_dir, class_name, source_manifest.get('source_kwargs', {}))
        for task_idx, task in enumerate(tasks, start=1):
            task_counts, task_rows = process_task(task)
            counts.update(task_counts)
            shard_rows.extend(task_rows)
            if task_idx % PROGRESS_EVERY_TASKS == 0:
                print('{} noise: {} tasks | samples {:,}'.format(dataset_name, task_idx, counts['samples_written']), flush=True)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(
            processes=NUM_WORKERS,
            initializer=init_worker,
            initargs=(dataset_name, str(dataset_path), str(out_dir), class_name, source_manifest.get('source_kwargs', {})),
        ) as pool:
            for task_idx, (task_counts, task_rows) in enumerate(pool.imap_unordered(process_task, tasks), start=1):
                counts.update(task_counts)
                shard_rows.extend(task_rows)
                if task_idx % PROGRESS_EVERY_TASKS == 0:
                    print('{} noise: {} tasks | samples {:,}'.format(dataset_name, task_idx, counts['samples_written']), flush=True)

    shard_index_path = fixed.write_shard_index(out_dir, shard_rows)
    sample_index_path = fixed.merge_index_parts(out_dir)
    failure_index_path = fixed.merge_failure_parts(out_dir)
    noise_sample_index_path = write_noise_metadata_file(out_dir / 'noise_sample_index.csv', sample_index_path)
    summary_path = out_dir / 'summary.json'
    summary = {
        'dataset': dataset_name,
        'status': 'ok',
        'input_noise_index': str(DATA_ROOT / dataset_name / NOISE_INDEX_SUBDIR / NOISE_INDEX_NAME),
        'dataset_path': str(dataset_path),
        'dataset_class': class_name,
        'source_kind': source_manifest.get('source_kind', ''),
        'selection_method': source_manifest.get('selection_method', ''),
        'source_kwargs': source_manifest.get('source_kwargs', {}),
        'out_dir': str(out_dir),
        'shard_index': str(shard_index_path),
        'sample_index': str(sample_index_path),
        'noise_sample_index': str(noise_sample_index_path),
        'failure_index': str(failure_index_path),
        'window_length_sec': WINDOW_LENGTH_SEC,
        'target_sample_rate': TARGET_SAMPLE_RATE,
        'noise_window_start_mode': NOISE_WINDOW_START_MODE,
        'random_seed': RANDOM_SEED,
        'num_input_rows': int(len(df)),
        'num_shards': int(len(shard_rows)),
        'counts': {key: int(val) for key, val in counts.items() if isinstance(val, (int, np.integer))},
    }
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print('{} noise done | samples {:,} | shards {:,}'.format(dataset_name, counts['samples_written'], len(shard_rows)), flush=True)
    return {
        'dataset': dataset_name,
        'status': 'ok',
        'num_input_rows': int(len(df)),
        'num_samples': int(counts['samples_written']),
        'num_shards': int(len(shard_rows)),
        'out_dir': str(out_dir),
        'shard_index': str(shard_index_path),
        'sample_index': str(sample_index_path),
        'noise_sample_index': str(noise_sample_index_path),
        'failure_index': str(failure_index_path),
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