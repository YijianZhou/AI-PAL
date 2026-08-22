"""Build a num_aug-aware PAL phase file from phase/station rarity."""
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, AND CASE IDENTITY
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()
CASE_CODE = "eg"
PHASE_FILE = Path("input/%s_pal_hyp.pha" % CASE_CODE)
STATION_FILE = Path("input/%s_station.csv" % CASE_CODE)
OUTPUT_DIR = Path("output/%s_phase_rarity" % CASE_CODE)
RARITY_PHASE_FILE = Path("input/%s_pal_rarity.pha" % CASE_CODE)

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
PAL_SRC = AI_PAL_ROOT / "PAL_src"
shutil.copyfile(
    "config_ai_pal_{}.py".format(CASE_CODE), PAL_SRC / "config_ai_pal.py"
)
sys.path.insert(0, str(PAL_SRC))
import config_ai_pal as config_module

cfg = config_module.Config()
if len(cfg.rarity_augmentation_values) != len(cfg.rarity_percentiles) + 1:
    raise ValueError(
        "rarity_augmentation_values must contain one more value than "
        "rarity_percentiles"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable, str(PAL_SRC / "phase_rarity.py"),
    "--phase-in", str(PHASE_FILE),
    "--station-file", str(STATION_FILE),
    "--training-pha-out", str(RARITY_PHASE_FILE),
    "--feature-csv-out", str(OUTPUT_DIR / "phase_feature_rarity.csv"),
    "--training-csv-out", str(OUTPUT_DIR / "phase_train_augmented.csv"),
    "--summary-out", str(OUTPUT_DIR / "phase_rarity_summary.csv"),
    "--feature-fig-out", str(OUTPUT_DIR / "phase_feature_distributions.jpg"),
    "--prob-fig-out", str(OUTPUT_DIR / "phase_rarity_distribution.jpg"),
    "--max-hypo-dist-km", str(cfg.rarity_max_hypo_dist_km),
    "--augmentation-values", *map(str, cfg.rarity_augmentation_values),
    "--rarity-percentiles", *map(str, cfg.rarity_percentiles),
    "--mag-bin-width", str(cfg.rarity_mag_bin_width),
    "--hypo-dist-bin-width", str(cfg.rarity_hypo_dist_bin_width),
    "--spatial-bin-km", str(cfg.rarity_spatial_bin_km),
    "--spatiotemporal-time-bin-days", str(cfg.rarity_time_bin_days),
]
subprocess.check_call(command)
