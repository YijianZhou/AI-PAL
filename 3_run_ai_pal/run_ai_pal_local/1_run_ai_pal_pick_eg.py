"""Run offline native AI pickers and build the preferred pick ensemble."""

from pathlib import Path
import sys


# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, AND TIME RANGE
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # Packaged example; drives configs, checkpoints, picks, and phase paths.
DATA_DIR = Path("/data/Example_data")
FULL_STATION_FILE = Path("input/example_pal_format4.sta")
TIME_RANGE = "20190704-20190707"  # Nominal dates; end date is exclusive.
NUM_WORKERS = 2  # Independent date-block processes; each loads its own models.
PREPROCESSING_WORKERS = 4  # Existing station I/O/preprocessing concurrency per process.
OVERWRITE_PICKS = False  # False resumes complete days and reruns incomplete days.
# Config paths are derived from CASE_CODE; picker selection is in config.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)
CKPT_ROOT = AI_PAL_ROOT / "Pre-trained_models" / "SoCal_2020-2025_ckpt"
PICK_ROOT = Path("output/%s" % CASE_CODE)
ENSEMBLE_PICK_DIR = PICK_ROOT / "1.2_picks_AI-PAL-ENSEMBLE"

# ============================================================================
# USER SETTINGS: AVAILABLE PICKERS, CHECKPOINTS, AND DEVICES
# Select Local and CEED Global models in config_ai_pal_<case>.py.
# Packaged checkpoints live under AI_PAL_ROOT / "Pre-trained_models".
# Set gpu_idx=-1 to run a picker on CPU.
# ============================================================================
PICKERS_LOCAL = {
    "SAR": {
        "config": Path("config_sar_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt": CKPT_ROOT / "sar_best.ckpt",
    },
    "FT": {
        "config": Path("config_ft_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt": CKPT_ROOT / "ft_best.ckpt",
    },
    "PHN": {
        "config": Path("config_phn_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt": CKPT_ROOT / "phn_best.ckpt",
    },
    "RUN": {
        "config": Path("config_run_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt": CKPT_ROOT / "run_best.ckpt",
    },
}

PICKERS_GLOBAL = {
    "SAR_CEED": {
        "config": Path("config_sar_global_ceed.py"),
        "gpu_idx": 0,
        "ckpt": AI_PAL_ROOT / "Pre-trained_models/CEED/CEED_ckpt/ceed_sar_best.ckpt",
    },
    "FT_CEED": {
        "config": Path("config_ft_global_ceed.py"),
        "gpu_idx": 0,
        "ckpt": AI_PAL_ROOT / "Pre-trained_models/CEED/CEED_ckpt/ceed_ft_best.ckpt",
    },
    "PHN_CEED": {
        "config": Path("config_phn_global_ceed.py"),
        "gpu_idx": 0,
        "ckpt": AI_PAL_ROOT / "Pre-trained_models/CEED/CEED_ckpt/ceed_phn_best.ckpt",
    },
    "RUN_CEED": {
        "config": Path("config_run_global_ceed.py"),
        "gpu_idx": 0,
        "ckpt": AI_PAL_ROOT / "Pre-trained_models/CEED/CEED_ckpt/ceed_run_best.ckpt",
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


for source_path in (AI_PAL_ROOT, PAL_SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from offline_pick_blocks import load_workflow_config, run_offline_pick_blocks


def main():
    workflow_cfg = load_workflow_config(case_path(CONFIG_AI_PAL))
    migrate_legacy_directory(
        PICK_ROOT / "picks_ENSEMBLE", ENSEMBLE_PICK_DIR
    )
    from continuous_pickers import continuous_specs
    picker_specs = continuous_specs(workflow_cfg, PICKERS_LOCAL, PICKERS_GLOBAL)
    for spec in picker_specs.values():
        spec["config"] = case_path(spec["config"])
        spec["ckpt"] = case_path(spec["ckpt"])
    print("selected continuous pickers: {}".format(list(picker_specs)), flush=True)
    run_offline_pick_blocks(
        ai_pal_root=AI_PAL_ROOT,
        config_path=case_path(CONFIG_AI_PAL),
        picker_specs=picker_specs,
        data_dir=DATA_DIR,
        station_file=case_path(FULL_STATION_FILE),
        time_range=TIME_RANGE,
        individual_pick_root=case_path(PICK_ROOT),
        ensemble_pick_dir=case_path(ENSEMBLE_PICK_DIR),
        num_workers=NUM_WORKERS,
        preprocessing_workers=PREPROCESSING_WORKERS,
        log_dir=case_path(PICK_ROOT / "pick_logs"),
        overwrite=OVERWRITE_PICKS,
    )


if __name__ == "__main__":
    main()
