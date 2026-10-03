"""Add annual local negatives to the CEED Zarr built by step 2.1."""
from pathlib import Path
import sys

# USER SETTINGS: replace these fictional example paths with real datasets.
AI_PAL_ROOT = Path('~/software/AI-PAL').expanduser()
LOCAL_ZARR = Path('/data/Example_data/eg-rarity-v1')
TRAINING_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
CEED_ZARR = Path('/data/Example_data/ceed_train-samples.zarr')
ENABLED_MODELS = ['SAR', 'FT', 'PHN', 'RUN']
BATCH_ROWS = 128  # Bounded copying memory; independent of training batch size.
NUM_WORKERS = 4  # Parallel chunk copies; use 1 for serial copying.
RESTART_LEGACY_TRANSFER = False  # True only after confirming the old copier stopped.
DATASET_SETTINGS_CONFIRMED = False  # True after replacing paths and years.


def main():
    if not DATASET_SETTINGS_CONFIRMED:
        raise ValueError('Replace example paths/years, then set DATASET_SETTINGS_CONFIRMED=True.')
    sys.path.insert(0, str(AI_PAL_ROOT / 'Pre-trained_models/CEED/helpers'))
    from ceed_negative_transfer import copy_annual_negatives
    copy_annual_negatives(LOCAL_ZARR, CEED_ZARR, TRAINING_YEARS,
                          ENABLED_MODELS, batch_rows=BATCH_ROWS,
                          restart_legacy=RESTART_LEGACY_TRANSFER,
                          num_workers=NUM_WORKERS)


if __name__ == '__main__':
    main()
