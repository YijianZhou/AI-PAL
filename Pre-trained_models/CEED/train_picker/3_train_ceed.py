"""Train CEED positives with local negatives from an integrated Zarr store."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: DATASET, MODELS, OUTPUTS, AND EXECUTION
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "ceed"
LOCAL_CASE_CODE = "eg"  # Placeholder local negative source; supply your own data.
CONFIG_AI_PAL = Path(__file__).resolve().parent.parent / "preprocess/config_ai_pal_ceed.py"
DATASET_SETTINGS_CONFIRMED = False  # Set True only after replacing placeholder paths.
ENABLED_MODELS = ["SAR", "FT", "PHN", "RUN"]
ZARR_PATH = Path("/data/Example_data/%s_train-samples.zarr" % CASE_CODE)
OUT_CKPT_ROOT = Path("output/%s_ckpt" % CASE_CODE)

# Models train sequentially, so they share one GPU assignment.
gpu_idx = 0
NUM_WORKERS = 10
PREFETCH_FACTOR = 2

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
if not DATASET_SETTINGS_CONFIRMED:
    raise ValueError("Optional CEED retraining: replace Example_data paths with real datasets, then set DATASET_SETTINGS_CONFIRMED=True.")
sys.path.insert(0, str(AI_PAL_ROOT / "PAL_src"))
# Dataset checks in each trainer follow that model's batch_size[1].
if (ZARR_PATH / '.negative_transfer_in_progress').exists():
    raise RuntimeError('Negative transfer is incomplete; do not train: ' + str(ZARR_PATH))

shutil.copyfile(
    CONFIG_AI_PAL,
    AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
)
subprocess_env = os.environ.copy()
subprocess_env.pop("AI_PAL_TRAINING_BLOCKS", None)
subprocess_env.pop("AI_PAL_TRAINING_YEARS", None)
subprocess_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
    str(AI_PAL_ROOT / "PAL_src"),
    subprocess_env.get("PYTHONPATH"),
)))

dataset_label = "CEED positives + local negatives"

print("training dataset: {} ({})".format(ZARR_PATH, dataset_label), flush=True)

for model_name in ENABLED_MODELS:
    model_code = model_name.lower()
    src_dir = AI_PAL_ROOT / "picker_{}".format(model_name)
    config_path = Path(__file__).resolve().parent / "config_{}_global_ceed.py".format(model_code)
    train_script = src_dir / "train.py"
    ckpt_dir = OUT_CKPT_ROOT / model_name

    for required_path in (src_dir, config_path, train_script):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, src_dir / "config.py")
    shutil.copyfile(
        CONFIG_AI_PAL,
        src_dir / "config_ai_pal.py",
    )
    print(
        "CEED + local-negative training {} from {} -> {}".format(
            model_name, dataset_label, ckpt_dir
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
    subprocess.check_call(command, env=subprocess_env)
