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
TIME_RANGE = "20190704-20190707"  # Exclusive end date.
NUM_WORKERS = 4  # Concurrent station-date I/O/preprocessing workers.
OVERWRITE_PICKS = False  # False resumes complete days and reruns incomplete days.
# Config paths are derived from CASE_CODE; picker selection is in config.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)
CKPT_ROOT = Path("output/%s_ckpt" % CASE_CODE)
PICK_ROOT = Path("output/%s" % CASE_CODE)
ENSEMBLE_PICK_DIR = PICK_ROOT / "1.2_picks_AI-PAL-ENSEMBLE"

# ============================================================================
# USER SETTINGS: AVAILABLE PICKERS, CHECKPOINTS, AND DEVICES
# Select both preferred groups in config_ai_pal_<case>.py. POS_NEG loads the
# latest .ckpt in each directory; POS uses the explicit checkpoint files below.
# Set gpu_idx=-1 to run a picker on CPU.
# ============================================================================
PICKERS_POS_NEG = {
    "SAR": {
        "config": Path("config_sar_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt_dir": CKPT_ROOT / "SAR",
    },
    "FT": {
        "config": Path("config_ft_%s.py" % CASE_CODE),
        "gpu_idx": 1,
        "ckpt_dir": CKPT_ROOT / "FT",
    },
    "PHN": {
        "config": Path("config_phn_%s.py" % CASE_CODE),
        "gpu_idx": 2,
        "ckpt_dir": CKPT_ROOT / "PHN",
    },
    "RUN": {
        "config": Path("config_run_%s.py" % CASE_CODE),
        "gpu_idx": 3,
        "ckpt_dir": CKPT_ROOT / "RUN",
    },
}

PICKERS_POS = {
    "SAR": {
        "config": Path("config_sar_pos_%s.py" % CASE_CODE),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_sar.ckpt"),
    },
    "FT": {
        "config": Path("config_ft_pos_%s.py" % CASE_CODE),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_ft.ckpt"),
    },
    "PHN": {
        "config": Path("config_phn_pos_%s.py" % CASE_CODE),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_phn-1m.ckpt"),
    },
    "RUN": {
        "config": Path("config_run_pos_%s.py" % CASE_CODE),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_run.ckpt"),
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
    selected_pos = list(dict.fromkeys(workflow_cfg.picker_pos_group))
    if not selected_pos_neg and not selected_pos:
        raise ValueError("at least one preferred continuous picker is required")
    missing_pos_neg = set(selected_pos_neg) - set(PICKERS_POS_NEG)
    missing_pos = set(selected_pos) - set(PICKERS_POS)
    if missing_pos_neg or missing_pos:
        raise ValueError(
            "selected picker specifications are missing: POS_NEG={} POS={}"
            .format(sorted(missing_pos_neg), sorted(missing_pos))
        )
    picker_specs = {}
    for name in selected_pos_neg:
        picker_specs[name] = {
            **PICKERS_POS_NEG[name],
            "group": "POS_NEG",
            "model": name,
            "config": case_path(PICKERS_POS_NEG[name]["config"]),
            "ckpt_dir": case_path(PICKERS_POS_NEG[name]["ckpt_dir"]),
        }
    for name in selected_pos:
        picker_specs["POS-{}".format(name)] = {
            **PICKERS_POS[name],
            "group": "POS",
            "model": name,
            "config": case_path(PICKERS_POS[name]["config"]),
            "ckpt": case_path(PICKERS_POS[name]["ckpt"]),
        }
    print(
        "selected preferred pickers: POS_NEG={} POS={}".format(
            selected_pos_neg, selected_pos
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
