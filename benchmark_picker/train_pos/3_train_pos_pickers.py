"""Train configured AI-PAL picker models with positive-event samples only."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import zarr

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


def validate_positive_zarr(zarr_path, enabled_models):
    target_names = {
        "SAR": "frame", "FT": "frame", "PHN": "sample", "RUN": "sample",
    }
    for split in ("train", "valid"):
        data_path = zarr_path / split / "positive_data"
        data = zarr.open(str(data_path), mode="r")
        if data.shape[0] <= 0:
            raise ValueError("empty positive dataset: {}".format(data_path))
        if len(data.shape) != 3 or data.shape[1] != 3:
            raise ValueError(
                "expected positive waveforms [sample, 3, time], got {} in {}".format(
                    data.shape, data_path
                )
            )
        for target_name in sorted({target_names[name] for name in enabled_models}):
            target_path = zarr_path / split / (
                "positive_target_{}".format(target_name)
            )
            target = zarr.open(str(target_path), mode="r")
            if target.shape[0] != data.shape[0]:
                raise ValueError(
                    "positive data/target count mismatch: {} versus {}".format(
                        data_path, target_path
                    )
                )
            if target_name == "frame" and len(target.shape) != 2:
                raise ValueError(
                    "expected frame targets [sample, frame], got {} in {}".format(
                        target.shape, target_path
                    )
                )
            if target_name == "sample" and (
                len(target.shape) != 3
                or target.shape[1] != 3
                or target.shape[2] != data.shape[2]
            ):
                raise ValueError(
                    "expected sample targets [sample, 3, time], got {} in {}".format(
                        target.shape, target_path
                    )
                )
        print(
            "validated {} positive samples: {:,}".format(split, data.shape[0]),
            flush=True,
        )



def main():
    unknown_models = set(ENABLED_MODELS) - {"SAR", "FT", "PHN", "RUN"}
    if unknown_models:
        raise KeyError("unknown models: {}".format(sorted(unknown_models)))
    if not ENABLED_MODELS or len(set(ENABLED_MODELS)) != len(ENABLED_MODELS):
        raise ValueError("ENABLED_MODELS must contain unique model names")
    if not ZARR_PATH.exists():
        raise FileNotFoundError(ZARR_PATH)
    validate_positive_zarr(ZARR_PATH, ENABLED_MODELS)
    for name in ENABLED_MODELS:
        for required in (
            AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
            SCRIPT_DIR / "config_ai_pal_pos.py",
            SCRIPT_DIR / ("config_{}_pos.py".format(name.lower())),
            AI_PAL_ROOT / ("picker_" + name) / "train.py",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        output = OUT_CKPT_ROOT / name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                "Training starts from scratch; choose an empty checkpoint directory: {}".format(output)
            )
    shutil.copyfile(
        SCRIPT_DIR / "config_ai_pal_pos.py",
        AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
    )
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(AI_PAL_ROOT / "PAL_src"),
        subprocess_env.get("PYTHONPATH"),
    )))

    for model_name in ENABLED_MODELS:
        model_code = model_name.lower()
        src_dir = AI_PAL_ROOT / "picker_{}".format(model_name)
        config_path = SCRIPT_DIR / "config_{}_pos.py".format(model_code)
        train_script = src_dir / "train.py"
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
        command.append("--positive_only")
        subprocess.check_call(command, env=subprocess_env)


if __name__ == "__main__":
    main()
