"""Postprocess offline PAL detections with positive repicking/reassociation."""

from pathlib import Path
import shutil
import sys


# ============================================================================
# USER SETTINGS: PACKAGE, INPUTS, OUTPUTS, AND TARGET TIME RANGE
# Run 2.1 and 2.2 first. This stage reads the daily initial detections written
# by 2.2 and writes one final postprocessed phase file per UTC day.
# Toggle event plots and filtered SAC output in config_ai_pal_<case>.py with
# enable_event_waveform_plot and save_filtered_event_waveforms.
# This dedicated stage requires enable_post_process=True in that config.
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()
CASE_CODE = "eg"
DATA_DIR = Path("/data/Example_data")
FULL_STATION_FILE = Path("input/example_pal_format1.sta")
TIME_RANGE = "20190704-20190707"  # Exclusive end date.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)

RESULT_ROOT = Path("output/%s" % CASE_CODE)
INITIAL_PHASE_DIR = (
    RESULT_ROOT / "2.1.0_phase_init_AI-PAL" / "daily_assoc" / "merged"
)
FINAL_ROOT = RESULT_ROOT / "3.1_phase_final_AI-PAL"


# ============================================================================
# USER SETTINGS: AVAILABLE REPICKERS, CHECKPOINTS, AND DEVICES
# This separate stage loads all eight models because no continuous models are
# resident to reuse. Copy best checkpoints into input and specify exact files.
# Set gpu_idx=-1 for CPU inference.
# ============================================================================
CKPT_ROOT = Path("input/%s_ckpt" % CASE_CODE)
PICKERS_POS_NEG = {
    name: {
        "config": Path("config_%s_%s.py" % (name.lower(), CASE_CODE)),
        "gpu_idx": -1,
        "ckpt": CKPT_ROOT / ("%s_best.ckpt" % name.lower()),
    }
    for name in ("SAR", "FT", "PHN", "RUN")
}
PICKERS_POS = {
    "SAR": {
        "config": Path("config_sar_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_sar.ckpt"),
    },
    "FT": {
        "config": Path("config_ft_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_ft.ckpt"),
    },
    "PHN": {
        "config": Path("config_phn_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_phn-1m.ckpt"),
    },
    "RUN": {
        "config": Path("config_run_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_run.ckpt"),
    },
}


# ============================================================================
# USER SETTINGS: EXECUTION
# Events are processed sequentially so all repicker models load only once.
# Workers concurrently read, merge, gain-correct, and filter station spans for
# the current event. OVERWRITE=False resumes completed daily phase files.
# Set it True after changing repick settings or enabling a new output product.
# ============================================================================
NUM_WORKERS = 4
OVERWRITE = False


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
from offline_event_postprocessor import run_offline_event_postprocessing


def main():
    workflow_cfg = cfg.Config()
    migrate_legacy_directory(
        RESULT_ROOT / "phase_ENSEMBLE_PAL",
        RESULT_ROOT / "2.1.0_phase_init_AI-PAL",
    )
    migrate_legacy_directory(
        RESULT_ROOT / "phase_ENSEMBLE_PAL_final", FINAL_ROOT
    )
    selected_pos_neg = list(dict.fromkeys(
        workflow_cfg.repicker_pos_neg_group
    ))
    selected_pos = list(dict.fromkeys(workflow_cfg.repicker_pos_group))
    pos_neg_specs = {
        name: {
            "config": case_path(PICKERS_POS_NEG[name]["config"]),
            "gpu_idx": PICKERS_POS_NEG[name]["gpu_idx"],
            "ckpt": case_path(PICKERS_POS_NEG[name]["ckpt"]),
        }
        for name in selected_pos_neg
    }
    pos_specs = {
        name: {
            **PICKERS_POS[name],
            "config": case_path(PICKERS_POS[name]["config"]),
            "ckpt": case_path(PICKERS_POS[name]["ckpt"]),
        }
        for name in selected_pos
    }
    print(
        "selected repickers: POS_NEG={} POS={}".format(
            selected_pos_neg, selected_pos
        ), flush=True,
    )
    run_offline_event_postprocessing(
        ai_pal_root=AI_PAL_ROOT,
        cfg=workflow_cfg,
        repicker_pos_neg_specs=pos_neg_specs,
        repicker_pos_specs=pos_specs,
        data_dir=case_path(DATA_DIR),
        station_file=case_path(FULL_STATION_FILE),
        initial_phase_dir=case_path(INITIAL_PHASE_DIR),
        final_root=case_path(FINAL_ROOT),
        time_range=TIME_RANGE,
        num_workers=NUM_WORKERS,
        overwrite=OVERWRITE,
    )


if __name__ == "__main__":
    main()
