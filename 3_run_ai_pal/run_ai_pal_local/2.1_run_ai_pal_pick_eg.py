"""Run offline native AI pickers and build the preferred pick ensemble."""

from pathlib import Path
import shutil
import sys


# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, AND TIME RANGE
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # Packaged example; drives configs, checkpoints, picks, and phase paths.
DATA_DIR = Path("/data/Example_data")
FULL_STATION_FILE = Path("input/example_pal_format1.sta")
TIME_RANGE = "20190704-20190707"  # Nominal dates; end date is exclusive.
NUM_WORKERS = 4  # Concurrent station-date I/O/preprocessing workers.
OVERWRITE_PICKS = False  # False resumes complete days and reruns incomplete days.
# Config paths are derived from CASE_CODE; picker selection is in config.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)
CKPT_ROOT = Path("input/%s_ckpt" % CASE_CODE)
PICK_ROOT = Path("output/%s" % CASE_CODE)
ENSEMBLE_PICK_DIR = PICK_ROOT / "1.2_picks_AI-PAL-ENSEMBLE"

# ============================================================================
# USER SETTINGS: AVAILABLE PICKERS, CHECKPOINTS, AND DEVICES
# Select POS_NEG models in config_ai_pal_<case>.py.
# Copy best training checkpoints into input; specify exact files below.
# Set gpu_idx=-1 to run a picker on CPU.
# ============================================================================
PICKERS_POS_NEG = {
    "SAR": {
        "config": Path("config_sar_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt": CKPT_ROOT / "sar_best.ckpt",
    },
    "FT": {
        "config": Path("config_ft_%s.py" % CASE_CODE),
        "gpu_idx": 1,
        "ckpt": CKPT_ROOT / "ft_best.ckpt",
    },
    "PHN": {
        "config": Path("config_phn_%s.py" % CASE_CODE),
        "gpu_idx": 2,
        "ckpt": CKPT_ROOT / "phn_best.ckpt",
    },
    "RUN": {
        "config": Path("config_run_%s.py" % CASE_CODE),
        "gpu_idx": 3,
        "ckpt": CKPT_ROOT / "run_best.ckpt",
    },
}

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
RUN_DIR = Path(__file__).resolve().parent
PAL_SRC = AI_PAL_ROOT / "PAL_src"


def case_path(path):
    return path if path.is_absolute() else RUN_DIR / path


def migrate_legacy_directory(legacy, current):
    legacy = case_path(legacy)
    current = case_path(current)
    if legacy.is_dir() and not current.exists():
        current.parent.mkdir(parents=True, exist_ok=True)
        legacy.replace(current)
        print("migrated legacy local output: {} -> {}".format(
            legacy, current
        ))


shutil.copyfile(case_path(CONFIG_AI_PAL), PAL_SRC / "config_ai_pal.py")
for source_path in (AI_PAL_ROOT, PAL_SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

import config_ai_pal as cfg
from offline_picker_runner import run_offline_picker_ensemble


def main():
    workflow_cfg = cfg.Config()
    migrate_legacy_directory(
        PICK_ROOT / "picks_ENSEMBLE", ENSEMBLE_PICK_DIR
    )
    selected_pos_neg = list(dict.fromkeys(
        workflow_cfg.picker_pos_neg_group
    ))
    if not selected_pos_neg:
        raise ValueError("at least one preferred continuous picker is required")
    missing_pos_neg = set(selected_pos_neg) - set(PICKERS_POS_NEG)
    if missing_pos_neg:
        raise ValueError(
            "selected picker specifications are missing: POS_NEG={}"
            .format(sorted(missing_pos_neg))
        )
    picker_specs = {}
    for name in selected_pos_neg:
        picker_specs[name] = {
            **PICKERS_POS_NEG[name],
            "group": "POS_NEG",
            "model": name,
            "config": case_path(PICKERS_POS_NEG[name]["config"]),
            "ckpt": case_path(PICKERS_POS_NEG[name]["ckpt"]),
        }
    print(
        "selected preferred pickers: POS_NEG={}".format(
            selected_pos_neg
        ),
        flush=True,
    )
    run_offline_picker_ensemble(
        ai_pal_root=AI_PAL_ROOT,
        cfg=workflow_cfg,
        picker_specs=picker_specs,
        data_dir=DATA_DIR,
        station_file=case_path(FULL_STATION_FILE),
        time_range=TIME_RANGE,
        individual_pick_root=case_path(PICK_ROOT),
        ensemble_pick_dir=case_path(ENSEMBLE_PICK_DIR),
        num_workers=NUM_WORKERS,
        overwrite=OVERWRITE_PICKS,
    )


if __name__ == "__main__":
    main()
