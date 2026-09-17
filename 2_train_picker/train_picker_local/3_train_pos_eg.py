"""Train positive-only models from integrated or annual Zarr stores."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: DATASET, MODELS, OUTPUTS, AND EXECUTION
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # "eg" is the packaged example; use e.g. "sc" for SoCal.
ENABLED_MODELS = ["SAR", "FT", "PHN", "RUN"]
ZARR_PATH = Path("/data/bigdata/%s_train-samples.zarr" % CASE_CODE)
# None selects the integrated Zarr at ZARR_PATH. For an annual archive, point
# ZARR_PATH at the directory containing <year>.zarr and list the desired years.
TRAINING_YEARS = None  # e.g. (2020, 2021, 2022)
OUT_CKPT_ROOT = Path("output/%s_ckpt_pos" % CASE_CODE)

# Models train sequentially, so they share one GPU assignment.
gpu_idx = 0
NUM_WORKERS = 10
PREFETCH_FACTOR = 2

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
shutil.copyfile(
    "config_ai_pal_{}.py".format(CASE_CODE),
    AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
)
subprocess_env = os.environ.copy()
subprocess_env.pop("AI_PAL_TRAINING_BLOCKS", None)
subprocess_env.pop("AI_PAL_TRAINING_YEARS", None)
subprocess_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
    str(AI_PAL_ROOT / "PAL_src"),
    subprocess_env.get("PYTHONPATH"),
)))

if not ZARR_PATH.exists():
    raise FileNotFoundError(ZARR_PATH)
if TRAINING_YEARS is None:
    if not (ZARR_PATH / "train" / "positive_data").exists():
        raise ValueError(
            "ZARR_PATH is not an integrated Zarr; set TRAINING_YEARS when "
            "using an annual archive"
        )
    dataset_label = "integrated"
else:
    if not TRAINING_YEARS:
        raise ValueError("TRAINING_YEARS cannot be empty; use None instead")
    if len(set(TRAINING_YEARS)) != len(TRAINING_YEARS):
        raise ValueError("TRAINING_YEARS contains duplicates")
    if tuple(sorted(TRAINING_YEARS)) != tuple(TRAINING_YEARS):
        raise ValueError("TRAINING_YEARS must be chronological")
    missing_stores = [
        ZARR_PATH / "{}.zarr".format(year)
        for year in TRAINING_YEARS
        if not (ZARR_PATH / "{}.zarr".format(year)).exists()
    ]
    if missing_stores:
        raise FileNotFoundError(
            "missing annual Zarr store(s): {}".format(
                ", ".join(map(str, missing_stores))
            )
        )
    subprocess_env["AI_PAL_TRAINING_YEARS"] = ",".join(
        map(str, TRAINING_YEARS)
    )
    dataset_label = "annual years {}".format(
        ", ".join(map(str, TRAINING_YEARS))
    )

print("training dataset: {} ({})".format(ZARR_PATH, dataset_label), flush=True)

for model_name in ENABLED_MODELS:
    model_code = model_name.lower()
    src_dir = AI_PAL_ROOT / "picker_{}".format(model_name)
    config_path = Path("config_{}_{}.py".format(model_code, CASE_CODE))
    train_script = src_dir / "train.py"
    ckpt_dir = OUT_CKPT_ROOT / model_name

    for required_path in (src_dir, config_path, train_script):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, src_dir / "config.py")
    shutil.copyfile(
        "config_ai_pal_{}.py".format(CASE_CODE),
        src_dir / "config_ai_pal.py",
    )
    print(
        "positive-only training {} from {} -> {}".format(
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
    command.append("--positive_only")
    subprocess.check_call(command, env=subprocess_env)
