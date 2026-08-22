#!/usr/bin/env python3
"""Container entry point for one model's GPU training job."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MODEL_DIR = Path("/opt/ml/processing/model")
ZARR_INPUT_ROOT = Path("/opt/ml/processing/input/zarr")
OUTPUT_DIR = Path("/opt/ml/processing/output/checkpoints")
REQUIREMENTS = MODEL_DIR / "requirements.txt"

MODEL_TARGET_TYPES = {
    "SAR": "frame",
    "FT": "frame",
    "PHN": "sample",
    "RUN": "sample",
}


def validate_training_zarr(zarr_path, model):
    """Validate the stored positive/negative arrays consumed by one model."""
    import zarr

    try:
        target_type = MODEL_TARGET_TYPES[model]
    except KeyError as exc:
        raise KeyError("unknown model: {}".format(model)) from exc

    root = zarr.open(str(zarr_path), mode="r")
    shapes = {}
    for split in ("train", "valid"):
        for sample_kind in ("positive", "negative"):
            data_name = "{}/{}_data".format(split, sample_kind)
            target_name = "{}/{}_target_{}".format(
                split, sample_kind, target_type
            )
            arrays = []
            for name in (data_name, target_name):
                try:
                    arrays.append(root[name])
                except KeyError:
                    raise KeyError("missing required Zarr array: {}".format(name))
            data, target = arrays
            if data.shape[0] == 0:
                raise ValueError("empty required Zarr array: {}".format(data_name))
            if target.shape[0] != data.shape[0]:
                raise ValueError(
                    "sample-count mismatch: {} {} vs {} {}".format(
                        data_name, data.shape, target_name, target.shape
                    )
                )
            shapes[data_name] = list(data.shape)
            shapes[target_name] = list(target.shape)
    return target_type, shapes


def main():
    if REQUIREMENTS.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )
    model = os.environ["MODEL_NAME"]
    training_mode = os.environ.get("TRAINING_MODE", "positive_negative")
    if training_mode != "positive_negative":
        raise ValueError(
            "AWS training requires TRAINING_MODE=positive_negative, got {}".format(
                training_mode
            )
        )
    train_script = os.environ.get("TRAIN_SCRIPT", "train.py")
    zarr_name = os.environ.get("ZARR_NAME", "shared.zarr")
    zarr_path = ZARR_INPUT_ROOT / zarr_name
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)
    script_path = MODEL_DIR / train_script
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    target_type, array_shapes = validate_training_zarr(zarr_path, model)
    print(
        "training mode: positive + negative | {} targets".format(target_type),
        flush=True,
    )

    resume_dir = Path("/opt/ml/processing/resume")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if resume_dir.exists():
        shutil.copytree(resume_dir, OUTPUT_DIR, dirs_exist_ok=True)

    workers = int(os.environ.get("NUM_WORKERS", "10"))
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "2"))
    command = [
        sys.executable,
        script_path,
        "--gpu_idx", "0",
        "--num_workers", workers,
        "--prefetch_factor", prefetch_factor,
        "--zarr_path", zarr_path,
        "--ckpt_dir", OUTPUT_DIR,
    ]
    print("training {}: {}".format(model, " ".join(map(str, command))), flush=True)
    started = time.perf_counter()
    subprocess.check_call([str(item) for item in command], cwd=str(MODEL_DIR))
    elapsed = time.perf_counter() - started
    checkpoint_count = len(list(OUTPUT_DIR.glob("*.ckpt")))
    manifest = {
        "model": model,
        "train_script": train_script,
        "training_mode": training_mode,
        "target_type": target_type,
        "array_shapes": array_shapes,
        "zarr_path": str(zarr_path),
        "num_workers": workers,
        "prefetch_factor": prefetch_factor,
        "checkpoint_count": checkpoint_count,
        "training_sec": elapsed,
    }
    (OUTPUT_DIR / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
