"""Build yearly num_aug-aware PAL phase files before AWS sample cutting."""
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: CASE, YEARS, AND PREPARED INPUTS
# ============================================================================
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()
CASE_CODE = "eg"
YEARS = (2020, 2021, 2022)
STATION_FILE = Path("input/station_scedc_aws_selected_20200101_20260701_pal.csv")

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

for year in YEARS:
    year_dir = Path("input") / str(year)
    phase_in = year_dir / ("%s_assoc_%d_pal.pha" % (CASE_CODE, year))
    phase_out = year_dir / ("%s_assoc_%d_pal_rarity.pha" % (CASE_CODE, year))
    output_dir = Path("output") / ("%s_%d_phase_rarity" % (CASE_CODE, year))
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(PAL_SRC / "phase_rarity.py"),
        "--phase-in", str(phase_in),
        "--station-file", str(STATION_FILE),
        "--training-pha-out", str(phase_out),
        "--feature-csv-out", str(output_dir / "phase_feature_rarity.csv"),
        "--training-csv-out", str(output_dir / "phase_train_augmented.csv"),
        "--summary-out", str(output_dir / "phase_rarity_summary.csv"),
        "--feature-fig-out", str(output_dir / "phase_feature_distributions.jpg"),
        "--prob-fig-out", str(output_dir / "phase_rarity_distribution.jpg"),
        "--max-hypo-dist-km", str(cfg.rarity_max_hypo_dist_km),
        "--augmentation-values", *map(str, cfg.rarity_augmentation_values),
        "--rarity-percentiles", *map(str, cfg.rarity_percentiles),
        "--mag-bin-width", str(cfg.rarity_mag_bin_width),
        "--hypo-dist-bin-width", str(cfg.rarity_hypo_dist_bin_width),
        "--spatial-bin-km", str(cfg.rarity_spatial_bin_km),
        "--spatiotemporal-time-bin-days", str(cfg.rarity_time_bin_days),
    ]
    print("analyzing phase rarity for {}".format(year), flush=True)
    subprocess.check_call(command)
