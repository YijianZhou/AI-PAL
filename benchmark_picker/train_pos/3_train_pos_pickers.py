"""Train configured AI-PAL picker models with positive-event samples only."""
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: DATASET, MODELS, OUTPUTS, AND EXECUTION
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
ENABLED_MODELS = ["SAR", "FT", "PHN", "RUN"]
ZARR_PATH = Path("/data1/zhouyj/CEED_train_pos.zarr")
OUT_CKPT_ROOT = Path("/nas/zhouyj/AI_ckpt/ceed_pos")

# Models train sequentially, so they share one GPU assignment.
gpu_idx = 0
NUM_WORKERS = 10
PREFETCH_FACTOR = 2

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
shutil.copyfile(
    SCRIPT_DIR / "config_ai_pal_pos.py",
    AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
)

if not ZARR_PATH.exists():
    raise FileNotFoundError(ZARR_PATH)

for model_name in ENABLED_MODELS:
    model_code = model_name.lower()
    src_dir = AI_PAL_ROOT / "picker_{}".format(model_name)
    config_path = SCRIPT_DIR / "config_{}_pos.py".format(model_code)
    # SAR retains its dedicated loop; other models use an explicit mode on
    # their standard trainer so normal training remains positive + negative.
    train_script = src_dir / ("train_pos.py" if model_name == "SAR" else "train.py")
    ckpt_dir = OUT_CKPT_ROOT / model_name

    for required_path in (src_dir, config_path, train_script):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, src_dir / "config.py")
    print(
        "positive-only training {} from {} -> {}".format(
            model_name, ZARR_PATH, ckpt_dir
        ),
        flush=True,
    )
    command = [
        sys.executable, str(train_script),
        "--gpu_idx", str(gpu_idx),
        "--num_workers", str(NUM_WORKERS),
        "--prefetch_factor", str(PREFETCH_FACTOR),
        "--zarr_path", str(ZARR_PATH),
        "--ckpt_dir", str(ckpt_dir),
    ]
    if model_name != "SAR":
        command.append("--positive_only")
    subprocess.check_call(command)
