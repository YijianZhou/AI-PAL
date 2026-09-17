#!/usr/bin/env python3
"""Container entry point for one model's GPU training job."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


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


def write_storage_status(model, stage):
    usage = shutil.disk_usage(OUTPUT_DIR)
    artifact_count = 0
    artifact_bytes = 0
    checkpoint_count = 0
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file() or path.name == "training_storage_status.json":
            continue
        artifact_count += 1
        artifact_bytes += path.stat().st_size
        checkpoint_count += path.suffix == ".ckpt"
    status = {
        "model": model,
        "stage": stage,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "filesystem_total_gib": usage.total / 1024 ** 3,
        "filesystem_used_gib": usage.used / 1024 ** 3,
        "filesystem_free_gib": usage.free / 1024 ** 3,
        "artifact_count": artifact_count,
        "artifact_bytes": artifact_bytes,
        "checkpoint_count": checkpoint_count,
    }
    path = OUTPUT_DIR / "training_storage_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def validate_training_zarr(zarr_root, model, years):
    """Validate every selected annual store consumed by one model."""
    import zarr

    try:
        target_type = MODEL_TARGET_TYPES[model]
    except KeyError as exc:
        raise KeyError("unknown model: {}".format(model)) from exc

    shapes = {}
    trailing_shapes = {}
    for year in years:
        zarr_path = zarr_root / "{}.zarr".format(year)
        if not zarr_path.exists():
            raise FileNotFoundError(zarr_path)
        year_shapes = {}
        for split in ("train", "valid"):
            for sample_kind in ("positive", "negative"):
                data_name = "{}/{}_data".format(split, sample_kind)
                target_name = "{}/{}_target_{}".format(
                    split, sample_kind, target_type
                )
                arrays = []
                for name in (data_name, target_name):
                    array_path = zarr_path / name
                    if not array_path.exists():
                        raise FileNotFoundError(
                            "{} missing required Zarr array: {}".format(year, name)
                        )
                    arrays.append(zarr.open(str(array_path), mode="r"))
                data, target = arrays
                if data.shape[0] == 0:
                    raise ValueError("{} has empty {}".format(year, data_name))
                if target.shape[0] != data.shape[0]:
                    raise ValueError(
                        "{} sample-count mismatch: {} {} vs {} {}".format(
                            year, data_name, data.shape, target_name, target.shape
                        )
                    )
                year_shapes[data_name] = list(data.shape)
                year_shapes[target_name] = list(target.shape)
                for name, array in ((data_name, data), (target_name, target)):
                    trailing = tuple(array.shape[1:])
                    if name in trailing_shapes and trailing_shapes[name] != trailing:
                        raise ValueError(
                            "{} shape differs across years: {} vs {}".format(
                                name, trailing_shapes[name], trailing
                            )
                        )
                    trailing_shapes[name] = trailing
        shapes[str(year)] = year_shapes
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
    years = tuple(
        int(value) for value in os.environ.get("TRAINING_YEARS", "").split(",")
        if value
    )
    if not years:
        raise ValueError("TRAINING_YEARS is required")
    zarr_path = Path(os.environ.get("ZARR_ROOT", str(ZARR_INPUT_ROOT)))
    script_path = MODEL_DIR / train_script
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    target_type, array_shapes = validate_training_zarr(zarr_path, model, years)
    print(
        "training mode: positive + negative | {} targets".format(target_type),
        flush=True,
    )

    resume_dir = Path("/opt/ml/processing/resume")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if resume_dir.exists():
        shutil.copytree(resume_dir, OUTPUT_DIR, dirs_exist_ok=True)

    minimum_free_gib = float(os.environ.get("MIN_CHECKPOINT_FREE_GB", "100"))
    storage = write_storage_status(model, "preflight")
    if storage["filesystem_free_gib"] < minimum_free_gib:
        raise RuntimeError(
            "checkpoint filesystem has {:.2f} GiB free; {:.2f} GiB required"
            .format(storage["filesystem_free_gib"], minimum_free_gib)
        )

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
    process = subprocess.Popen([str(item) for item in command], cwd=str(MODEL_DIR))
    while process.poll() is None:
        storage = write_storage_status(model, "training")
        if storage["filesystem_free_gib"] < minimum_free_gib:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError(
                "checkpoint filesystem fell to {:.2f} GiB free; "
                "stopped at the configured {:.2f} GiB floor".format(
                    storage["filesystem_free_gib"], minimum_free_gib
                )
            )
        time.sleep(60)
    if process.returncode:
        write_storage_status(model, "failed")
        raise subprocess.CalledProcessError(process.returncode, command)
    elapsed = time.perf_counter() - started
    write_storage_status(model, "completed")
    checkpoint_count = len(list(OUTPUT_DIR.glob("*.ckpt")))
    manifest = {
        "model": model,
        "train_script": train_script,
        "training_mode": training_mode,
        "target_type": target_type,
        "array_shapes": array_shapes,
        "training_years": list(years),
        "zarr_archive": os.environ.get("ZARR_ARCHIVE"),
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
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "training_failure.txt").write_text(
            failure, encoding="utf-8"
        )
        raise
