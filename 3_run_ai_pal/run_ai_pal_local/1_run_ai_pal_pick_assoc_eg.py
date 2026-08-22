"""Run offline native AI picking and rolling buffered PAL association."""

from pathlib import Path
import shutil
import sys


# ============================================================================
# USER SETTINGS: PACKAGE, DATA, STATIONS, OUTPUTS, AND TARGET TIME RANGE
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()
CASE_CODE = "eg"
DATA_DIR = Path("/data/Example_data")
# Set to None to pick the selector-deduplicated union of all subnet files.
FULL_STATION_FILE = Path("input/example_pal_format1.sta")
# Leave empty for association with a provided full file. Otherwise only these
# subnets are associated in r1, r2, ... order; they are also the required
# picking-union inputs when FULL_STATION_FILE is None.
SUBNET_STATION_FILES = [
    # Path("input/station_r1.csv"),
    # Path("input/station_r2.csv"),
]
TIME_RANGE = "20190704-20190707"  # Target dates; exclusive end date.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)

CKPT_ROOT = Path("output/%s_ckpt" % CASE_CODE)
RESULT_ROOT = Path("output/%s" % CASE_CODE)
ENSEMBLE_PICK_DIR = RESULT_ROOT / "1.2_picks_AI-PAL-ENSEMBLE"
PHASE_ROOT = RESULT_ROOT / "2.1.0_phase_init_AI-PAL"
ASSOC_ROOT = PHASE_ROOT / "hourly_assoc"
FINAL_ROOT = RESULT_ROOT / "3.1_phase_final_AI-PAL"
OUTPUT_CATALOG = FINAL_ROOT / ("catalog_%s.dat" % TIME_RANGE)
OUTPUT_PHASE = FINAL_ROOT / ("phase_%s.dat" % TIME_RANGE)


# ============================================================================
# USER SETTINGS: AVAILABLE PICKERS, CHECKPOINTS, AND DEVICES
# Select continuous and both repicker groups in config_ai_pal_<case>.py.
# Continuous pickers automatically load the latest .ckpt in each ckpt_dir.
# Positive pickers intentionally use an exact checkpoint file path.
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

# Positive-only checkpoint registry used by event postprocessing.
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
# USER SETTINGS: EXECUTION
# Daily streams can be large; tune picking workers with host RAM in mind.
# NUM_ASSOC_WORKERS controls concurrently associated hourly intervals.
# ============================================================================
NUM_PICK_WORKERS = 4
NUM_ASSOC_WORKERS = 5
OVERWRITE_PICKS = False  # False resumes complete days and reruns incomplete days.


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
RUN_DIR = Path(__file__).resolve().parent
PAL_SRC = AI_PAL_ROOT / "PAL_src"


def case_path(path):
    return path if path.is_absolute() else RUN_DIR / path


def latest_checkpoint(directory):
    checkpoints = list(case_path(directory).glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError("no .ckpt files in {}".format(directory))
    return max(checkpoints, key=lambda path: (path.stat().st_mtime, path.name))


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
from offline_pick_assoc_runner import run_offline_pick_assoc
from station_sets import association_station_file_mapping, build_station_union


def main():
    workflow_cfg = cfg.Config()
    migrate_legacy_directory(
        RESULT_ROOT / "picks_ENSEMBLE", ENSEMBLE_PICK_DIR
    )
    migrate_legacy_directory(
        RESULT_ROOT / "phase_ENSEMBLE_PAL", PHASE_ROOT
    )
    migrate_legacy_directory(
        RESULT_ROOT / "phase_ENSEMBLE_PAL_final", FINAL_ROOT
    )
    subnet_station_files = [case_path(path) for path in SUBNET_STATION_FILES]
    if FULL_STATION_FILE is None:
        full_station_file = build_station_union(
            subnet_station_files,
            case_path(RESULT_ROOT) / "_internal" / "stations" / "picking_union.sta",
        )
        print(
            "FULL_STATION_FILE=None: picking station union written to {}"
            .format(full_station_file)
        )
    else:
        full_station_file = case_path(FULL_STATION_FILE)
    association_station_files = association_station_file_mapping(
        workflow_cfg, full_station_file, subnet_station_files
    )
    for station_file in association_station_files.values():
        if not station_file.exists():
            raise FileNotFoundError(station_file)

    selected_picker_pos_neg = list(dict.fromkeys(
        workflow_cfg.picker_pos_neg_group
    ))
    selected_picker_pos = list(dict.fromkeys(
        workflow_cfg.picker_pos_group
    ))
    if not selected_picker_pos_neg and not selected_picker_pos:
        raise ValueError("at least one preferred continuous picker is required")
    missing_picker_pos_neg = (
        set(selected_picker_pos_neg) - set(PICKERS_POS_NEG)
    )
    missing_picker_pos = set(selected_picker_pos) - set(PICKERS_POS)
    if missing_picker_pos_neg or missing_picker_pos:
        raise ValueError(
            "selected picker specifications are missing: POS_NEG={} POS={}"
            .format(
                sorted(missing_picker_pos_neg), sorted(missing_picker_pos)
            )
        )
    picker_specs = {}
    for name in selected_picker_pos_neg:
        picker_specs[name] = {
            **PICKERS_POS_NEG[name],
            "group": "POS_NEG",
            "model": name,
            "config": case_path(PICKERS_POS_NEG[name]["config"]),
            "ckpt_dir": case_path(PICKERS_POS_NEG[name]["ckpt_dir"]),
        }
    for name in selected_picker_pos:
        picker_specs["POS-{}".format(name)] = {
            **PICKERS_POS[name],
            "group": "POS",
            "model": name,
            "config": case_path(PICKERS_POS[name]["config"]),
            "ckpt": case_path(PICKERS_POS[name]["ckpt"]),
        }
    selected_pos_neg = list(dict.fromkeys(
        workflow_cfg.repicker_pos_neg_group
    ))
    selected_pos = list(dict.fromkeys(workflow_cfg.repicker_pos_group))
    missing_pos_neg = set(selected_pos_neg) - set(PICKERS_POS_NEG)
    missing_pos = set(selected_pos) - set(PICKERS_POS)
    if missing_pos_neg or missing_pos:
        raise ValueError(
            "selected repicker specifications are missing: POS_NEG={} POS={}"
            .format(sorted(missing_pos_neg), sorted(missing_pos))
        )
    if set(selected_picker_pos_neg) - set(selected_pos_neg):
        raise ValueError(
            "continuous POS_NEG pickers must be selected for postprocessing"
        )
    if set(selected_picker_pos) - set(selected_pos):
        raise ValueError(
            "continuous POS pickers must be selected for postprocessing"
        )
    repicker_pos_neg_specs = {
        name: {
            "config": case_path(PICKERS_POS_NEG[name]["config"]),
            "gpu_idx": PICKERS_POS_NEG[name]["gpu_idx"],
            "ckpt": latest_checkpoint(
                PICKERS_POS_NEG[name]["ckpt_dir"]
            ),
        }
        for name in selected_pos_neg
    }
    repicker_pos_specs = {
        name: {
            **PICKERS_POS[name],
            "config": case_path(PICKERS_POS[name]["config"]),
            "ckpt": case_path(PICKERS_POS[name]["ckpt"]),
        }
        for name in selected_pos
    }
    phase_root = case_path(PHASE_ROOT)
    final_root = case_path(FINAL_ROOT)
    phase_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    print("association station sets: {}".format({
        name: str(path) for name, path in association_station_files.items()
    }))
    print(
        "selected pickers: continuous POS_NEG={} POS={} | "
        "postprocess POS_NEG={} POS={}".format(
            selected_picker_pos_neg, selected_picker_pos,
            selected_pos_neg, selected_pos
        ),
        flush=True,
    )
    run_offline_pick_assoc(
        ai_pal_root=AI_PAL_ROOT,
        cfg=workflow_cfg,
        picker_specs=picker_specs,
        data_dir=DATA_DIR,
        full_station_file=full_station_file,
        subnet_station_files=association_station_files,
        target_time_range=TIME_RANGE,
        individual_pick_root=case_path(RESULT_ROOT),
        ensemble_pick_dir=case_path(ENSEMBLE_PICK_DIR),
        assoc_root=case_path(ASSOC_ROOT),
        final_root=final_root,
        output_catalog=case_path(OUTPUT_CATALOG),
        output_phase=case_path(OUTPUT_PHASE),
        num_pick_workers=NUM_PICK_WORKERS,
        num_assoc_workers=NUM_ASSOC_WORKERS,
        repicker_pos_neg_specs=repicker_pos_neg_specs,
        repicker_pos_specs=repicker_pos_specs,
        overwrite_picks=OVERWRITE_PICKS,
    )


if __name__ == "__main__":
    main()
