"""Associate the canonical offline picker ensemble with PAL."""

from pathlib import Path
import shutil
import sys


# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, AND TIME RANGE
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # Packaged example; drives the workflow config, picks, and phase paths.
PICK_NAME = "ENSEMBLE"
# May be None when SUBNET_STATION_FILES is not empty.
FULL_STATION_FILE = Path("input/example_pal_format1.sta")
# Leave empty for full-network association. Otherwise only these subnets are
# associated; their order maps to r1, r2, ... in CONFIG_AI_PAL.
SUBNET_STATION_FILES = [
    # Path("input/station_r1.csv"),
    # Path("input/station_r2.csv"),
]
PICK_ROOT = Path("output/%s" % CASE_CODE)
OUT_ROOT = Path("output/%s" % CASE_CODE)
TIME_RANGE = "20190704-20190707"  # Exclusive end date.
# The shared workflow config is derived from CASE_CODE.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)

# ============================================================================
# USER SETTINGS: EXECUTION
# ============================================================================
NUM_WORKERS = 3
OVERWRITE = False


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
RUN_DIR = Path(__file__).resolve().parent
PAL_SRC = AI_PAL_ROOT / "PAL_src"


def case_path(path):
    return path if path.is_absolute() else RUN_DIR / path


shutil.copyfile(case_path(CONFIG_AI_PAL), PAL_SRC / "config_ai_pal.py")
sys.path.insert(0, str(PAL_SRC))

import config_ai_pal as cfg
from association_runner import run_buffered_association
from station_sets import association_station_file_mapping


def main():
    workflow_cfg = cfg.Config()
    full_station_file = (
        case_path(FULL_STATION_FILE)
        if FULL_STATION_FILE is not None else None
    )
    subnet_station_files = [case_path(path) for path in SUBNET_STATION_FILES]
    association_station_files = association_station_file_mapping(
        workflow_cfg, full_station_file, subnet_station_files
    )
    for station_file in association_station_files.values():
        if not station_file.exists():
            raise FileNotFoundError(station_file)
    print("association station sets: {}".format({
        name: str(path) for name, path in association_station_files.items()
    }))

    pick_dir = case_path(PICK_ROOT) / "picks_{}".format(PICK_NAME)
    out_root = case_path(OUT_ROOT) / "phase_{}_PAL".format(PICK_NAME)
    out_root.mkdir(parents=True, exist_ok=True)
    run_buffered_association(
        subnet_station_files=association_station_files,
        pick_dir=pick_dir,
        assoc_root=out_root / "daily_assoc",
        time_range=TIME_RANGE,
        num_workers=NUM_WORKERS,
        config_factory=cfg.Config,
        overwrite=OVERWRITE,
        output_catalog=out_root / "catalog_{}.dat".format(TIME_RANGE),
        output_phase=out_root / "phase_{}.dat".format(TIME_RANGE),
    )


if __name__ == "__main__":
    main()
