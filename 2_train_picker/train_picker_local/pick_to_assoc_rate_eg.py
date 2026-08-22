"""Build one training association-rate CSV from legacy PAL results."""

from pathlib import Path
import subprocess
import sys


# ============================================================================
# USER SETTINGS: INPUTS, OUTPUT, AND MATCHING
# Both input files should cover the same complete study period.
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # Packaged example; one identifier for all case-specific input/output names.
PICK_FILE = Path("input/%s_full.pick" % CASE_CODE)
PHASE_FILE = Path("input/%s_full.pha" % CASE_CODE)
OUTPUT_FILE = Path("input/%s_association_rates.csv" % CASE_CODE)

# Match both P and S arrivals. "auto" accepts legacy PAL and newer AI picks.
MATCH_TOLERANCE_SEC = 0.1
PICK_FORMAT = "auto"  # auto, pal, or ai


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
subprocess.check_call([
    sys.executable,
    str(AI_PAL_ROOT / "PAL_src" / "pick_to_assoc_rate.py"),
    "--pick_file", str(PICK_FILE),
    "--phase_file", str(PHASE_FILE),
    "--output_file", str(OUTPUT_FILE),
    "--tolerance_sec", str(MATCH_TOLERANCE_SEC),
    "--pick_format", PICK_FORMAT,
])
