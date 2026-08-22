"""Evaluate configured model runs on positive and noise benchmark datasets."""
from pathlib import Path
import importlib

import evaluate_noise_false_detections as noise_evaluator
import evaluate_pos_picker_predictions as pos_evaluator
from benchmark_runs import NOISE_DATASETS, POSITIVE_DATASETS, SAMPLE_TYPES, enabled_runs


# =============================================================================
# USER SETTINGS
# =============================================================================
DATA_ROOT = Path('/nas/zhouyj/AI_datasets')
OUTPUT_ROOT = Path('output')
POSITIVE_EVAL_ROOT = OUTPUT_ROOT / 'picker_eval'
NOISE_EVAL_ROOT = OUTPUT_ROOT / 'picker_eval_noise'


def prediction_path(dataset, run, sample_type):
    return OUTPUT_ROOT / '{}_{}_{}_predictions.csv'.format(
        dataset, run['marker'], 'pos' if sample_type == 'positive' else 'noise'
    )


def evaluate_positive(dataset, run):
    subdir = 'npy_shards_fixed50' if dataset == 'CEED' else 'npy_shards_fixed40'
    marker = run['marker']
    pos_evaluator.DATASET_NAME = dataset
    pos_evaluator.SHARD_SUBDIR = subdir
    pos_evaluator.GT_INDEX = DATA_ROOT / dataset / subdir / 'cleaned_phase_index.csv'
    pos_evaluator.PREDICTION_FILES = {marker: prediction_path(dataset, run, 'positive')}
    pos_evaluator.OUT_DIR = POSITIVE_EVAL_ROOT / dataset / marker
    print('=' * 80, flush=True)
    print('positive evaluation: {} {}'.format(dataset, run['label']), flush=True)
    pos_evaluator.main()


def evaluate_noise(dataset, run):
    marker = run['marker']
    noise_evaluator.DATASET_NAME = dataset
    noise_evaluator.SHARD_SUBDIR = 'npy_noise_shards_fixed50'
    noise_evaluator.NOISE_INDEX = (
        DATA_ROOT / dataset / 'npy_noise_shards_fixed50' / 'noise_sample_index.csv'
    )
    noise_evaluator.PREDICTION_FILES = {marker: prediction_path(dataset, run, 'noise')}
    noise_evaluator.OUT_DIR = NOISE_EVAL_ROOT / dataset / marker
    noise_evaluator.FILTER_PREDICTIONS_TO_NOISE_INDEX = True
    print('=' * 80, flush=True)
    print('noise evaluation: {} {}'.format(dataset, run['label']), flush=True)
    noise_evaluator.main()


def main():
    importlib.reload(pos_evaluator)
    importlib.reload(noise_evaluator)
    runs = enabled_runs()
    if 'positive' in SAMPLE_TYPES:
        for dataset in POSITIVE_DATASETS:
            for run in runs:
                evaluate_positive(dataset, run)
    if 'noise' in SAMPLE_TYPES:
        for dataset in NOISE_DATASETS:
            for run in runs:
                evaluate_noise(dataset, run)


if __name__ == '__main__':
    main()
