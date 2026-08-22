#!/usr/bin/env python3
"""Fill gaps in a PAL station file's time-varying gain intervals."""

import os
import sys
from pathlib import Path

# ============================================================================
# USER SETTINGS: SOURCE PACKAGE, INPUT, OUTPUT, AND STUDY COVERAGE
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()
CASE_CODE = "eg"
INPUT_STATION_FILE = Path("input/%s_station_raw.csv" % CASE_CODE)
OUTPUT_STATION_FILE = Path("input/%s_station.csv" % CASE_CODE)
AUDIT_FILE = Path("output/%s/station_gain_interval_audit.csv" % CASE_CODE)
STUDY_START = None  # Example: "2010-01-01"; extends the first gain backward.
STUDY_END = None  # Exclusive; example: "2026-01-01".


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
RUN_DIR = Path(__file__).resolve().parent
PAL_DIR = Path(os.environ.get("PAL_DIR", str(AI_PAL_ROOT / "PAL_src")))
sys.path.insert(0, str(PAL_DIR))

from data_pipeline import normalize_station_gain_intervals


def case_path(path):
    return path if path.is_absolute() else RUN_DIR / path


def main():
    summary = normalize_station_gain_intervals(
        input_path=case_path(INPUT_STATION_FILE),
        output_path=case_path(OUTPUT_STATION_FILE),
        audit_path=case_path(AUDIT_FILE),
        coverage_start=STUDY_START,
        coverage_end=STUDY_END,
    )
    print("normalized station gain intervals: {}".format(summary))
    print("station file: {}".format(case_path(OUTPUT_STATION_FILE)))
    print("audit file: {}".format(case_path(AUDIT_FILE)))


if __name__ == "__main__":
    main()
