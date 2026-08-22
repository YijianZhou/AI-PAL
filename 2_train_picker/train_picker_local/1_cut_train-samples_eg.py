"""Cut shared continuous-waveform training samples into NPY shards."""
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, AND EXECUTION
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # "eg" is the packaged example; use e.g. "sc" for SoCal.
DATA_DIR = Path("/data/Example_data")
# Step 0 writes this file when positive_num_aug_mode == "phase". For the
# legacy fixed strategy, point this at the original PAL phase file instead.
PHASE_FILE = Path("input/%s_pal_rarity.pha" % CASE_CODE)
ASSOCIATION_RATE_FILE = Path("input/%s_association_rates.csv" % CASE_CODE)
NPY_ROOT = Path("/data/bigdata/%s_train-samples_npy" % CASE_CODE)
NUM_WORKERS = 10
SHARD_SIZE = 1024

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
PICKER_SRC = AI_PAL_ROOT / "picker_SAR"
PREPROCESS_SRC = PICKER_SRC / "preprocess"
# All models consume this common waveform/sample inventory. Model-specific
# labels are generated later by 2_npy2zarr_eg.py.
shutil.copyfile(
    "config_ai_pal_{}.py".format(CASE_CODE),
    AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
)
subprocess.check_call([
    sys.executable, str(PREPROCESS_SRC / "cut_positive_npy.py"),
    "--data_dir", str(DATA_DIR),
    "--fpha", str(PHASE_FILE),
    "--out_root", str(NPY_ROOT),
    "--num_workers", str(NUM_WORKERS),
    "--shard_size", str(SHARD_SIZE),
])
subprocess.check_call([
    sys.executable, str(PREPROCESS_SRC / "cut_negative_npy.py"),
    "--data_dir", str(DATA_DIR),
    "--fpha", str(PHASE_FILE),
    "--fassoc_rate", str(ASSOCIATION_RATE_FILE),
    "--out_root", str(NPY_ROOT),
    "--num_workers", str(NUM_WORKERS),
    "--shard_size", str(SHARD_SIZE),
])
