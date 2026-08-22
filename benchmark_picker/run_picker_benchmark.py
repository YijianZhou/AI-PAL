"""Run configured AI-PAL checkpoint versions on fixed benchmark NPY shards."""
from pathlib import Path
import json
import re
import subprocess
import sys

from benchmark_runs import NOISE_DATASETS, POSITIVE_DATASETS, SAMPLE_TYPES, enabled_runs


# =============================================================================
# USER SETTINGS: PATHS AND INFERENCE
# =============================================================================
AI_PAL_ROOT = Path('~/software/AI-PAL').expanduser()
DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
OUTPUT_ROOT = Path('output')

GPU_IDX = 0
NUM_WORKERS = 10
PREFETCH_FACTOR = 2
CKPT_IDX = -1  # Used only when a run's checkpoint points to a directory.
PICKER_BATCH_SIZE = 256
START_RANGE_SEC = [0.0, 10.0]
NUM_REPEAT = 20
RANDOM_SEED = 20250708
MIN_CLUSTER_SIZE = 1
OVERWRITE_COMPLETE = False


def prediction_path(dataset, run, sample_type):
    return OUTPUT_ROOT / '{}_{}_{}_predictions.csv'.format(
        dataset, run['marker'], 'pos' if sample_type == 'positive' else 'noise'
    )


def output_is_complete(path):
    summary_path = Path(str(path) + '.summary.json')
    if not path.exists() or not summary_path.exists():
        return False
    try:
        with summary_path.open('r') as fp:
            return bool(json.load(fp).get('complete', False))
    except (OSError, ValueError):
        return False


def dataset_inputs(dataset, sample_type):
    if sample_type == 'positive':
        if dataset not in POSITIVE_DATASETS:
            raise ValueError('Unsupported positive dataset: {}'.format(dataset))
        subdir = 'npy_shards_fixed50' if dataset == 'CEED' else 'npy_shards_fixed40'
        index_name = 'cleaned_phase_index.csv'
    elif sample_type == 'noise':
        if dataset not in NOISE_DATASETS:
            raise ValueError('Unsupported noise dataset: {}'.format(dataset))
        subdir = 'npy_noise_shards_fixed50'
        index_name = 'noise_sample_index.csv'
    else:
        raise ValueError('sample_type must be positive or noise')
    root = DATA_ROOT / dataset / subdir
    return root / 'pos.npy', root / index_name


def run_one(dataset, run, sample_type):
    model = run['model'].upper()
    marker = run['marker']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', marker):
        raise ValueError('Run marker is not filesystem-safe: {}'.format(marker))
    model_dir = AI_PAL_ROOT / 'picker_{}'.format(model)
    runner = model_dir / 'run_picker_pos.py'
    config_path = model_dir / 'config.py'
    checkpoint = Path(run['checkpoint']).expanduser()
    shard_index, sample_index = dataset_inputs(dataset, sample_type)
    out_file = prediction_path(dataset, run, sample_type)

    for path in (runner, config_path, checkpoint, shard_index, sample_index):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_is_complete(out_file) and not OVERWRITE_COMPLETE:
        print('skip complete output: {}'.format(out_file), flush=True)
        return

    print(
        '{} {} | {} | checkpoint {} | installed config {}'.format(
            dataset, sample_type, run['label'], checkpoint, config_path
        ),
        flush=True,
    )
    command = [
        sys.executable, str(runner),
        '--gpu_idx', str(GPU_IDX),
        '--num_workers', str(NUM_WORKERS),
        '--prefetch_factor', str(PREFETCH_FACTOR),
        '--shard_index', str(shard_index),
        '--sample_index', str(sample_index),
        '--out_file', str(out_file),
        '--ckpt_dir', str(checkpoint),
        '--ckpt_idx', str(CKPT_IDX),
        '--pos_start_min', str(START_RANGE_SEC[0]),
        '--pos_start_max', str(START_RANGE_SEC[1]),
        '--pos_num_repeat', str(NUM_REPEAT),
        '--pos_batch_size', str(PICKER_BATCH_SIZE),
        '--pos_random_seed', str(RANDOM_SEED),
        '--pos_min_cluster_size', str(MIN_CLUSTER_SIZE),
    ]
    subprocess.check_call(command)


def main():
    runs = enabled_runs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sample_type in SAMPLE_TYPES:
        datasets = POSITIVE_DATASETS if sample_type == 'positive' else NOISE_DATASETS
        for dataset in datasets:
            for run in runs:
                print('=' * 80, flush=True)
                run_one(dataset, run, sample_type)


if __name__ == '__main__':
    main()
