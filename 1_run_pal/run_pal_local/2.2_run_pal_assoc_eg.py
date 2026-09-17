#!/usr/bin/env python3
"""Run independent-day PAL association for existing local picks."""

import importlib
import os
import sys
from pathlib import Path

# ============================================================================
# USER SETTINGS: INPUTS AND OUTPUTS
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # Packaged example; drives the case config and all output paths.
# Use {"full": ...} for one network, or r1/r2/... keys in its case config.
SUBNET_STATION_FILES = {
    "full": Path("input/example_pal_format1.sta"),
}
PICK_DIR = Path("output/%s/picks" % CASE_CODE)
OUT_ROOT = Path("output/%s" % CASE_CODE)
TIME_RANGE = "20190704-20190707"  # Exclusive end date.

# ============================================================================
# USER SETTINGS: EXECUTION
# ============================================================================
NUM_WORKERS = 3
OVERWRITE = False
RETRY_FAILED_DAYS = True


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
RUN_DIR = Path(__file__).resolve().parent
PAL_DIR = Path(
    os.environ.get("PAL_DIR", str(AI_PAL_ROOT / "PAL_src"))
)
sys.path.insert(0, str(PAL_DIR))

cfg = importlib.import_module("config_%s" % CASE_CODE)
from association_runner import run_buffered_association


def case_path(path):
    return path if path.is_absolute() else RUN_DIR / path


def main():
    out_root = case_path(OUT_ROOT)
    run_buffered_association(
        subnet_station_files={
            name: case_path(path) for name, path in SUBNET_STATION_FILES.items()
        },
        pick_dir=case_path(PICK_DIR),
        assoc_root=out_root / "association",
        time_range=TIME_RANGE,
        num_workers=NUM_WORKERS,
        config_factory=cfg.Config,
        overwrite=OVERWRITE,
        retry_failed_days=RETRY_FAILED_DAYS,
        association_buffer_enabled=False,
        output_catalog=out_root / "catalog_{}.dat".format(TIME_RANGE),
        output_phase=out_root / "phase_{}.dat".format(TIME_RANGE),
    )


if __name__ == "__main__":
    main()
