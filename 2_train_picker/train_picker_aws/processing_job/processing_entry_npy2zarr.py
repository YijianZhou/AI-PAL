#!/usr/bin/env python3
"""Container entry point for model-specific NPY-to-Zarr conversion."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


SOURCE_ROOT = Path("/opt/ml/processing/source")
NPY_ROOT = Path("/opt/ml/processing/input/npy")
OUTPUT_ROOT = Path("/opt/ml/processing/output/zarr")
REQUIREMENTS = SOURCE_ROOT / "requirements.txt"
COMBINED_NPY_ROOT = Path("/opt/ml/processing/work/npy_combined")


def validate_shared_zarr(zarr_path, target_builders):
    """Require stored positive and negative examples for every target type."""
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    shapes = {}
    for split in ("train", "valid"):
        for sample_kind in ("positive", "negative"):
            data_name = "{}/{}_data".format(split, sample_kind)
            try:
                data = root[data_name]
            except KeyError:
                raise KeyError("missing required Zarr array: {}".format(data_name))
            if data.shape[0] == 0:
                raise ValueError("empty required Zarr array: {}".format(data_name))
            shapes[data_name] = list(data.shape)

            for target_type in target_builders:
                target_name = "{}/{}_target_{}".format(
                    split, sample_kind, target_type
                )
                try:
                    target = root[target_name]
                except KeyError:
                    raise KeyError(
                        "missing required Zarr array: {}".format(target_name)
                    )
                if target.shape[0] != data.shape[0]:
                    raise ValueError(
                        "sample-count mismatch: {} {} vs {} {}".format(
                            data_name, data.shape, target_name, target.shape
                        )
                    )
                shapes[target_name] = list(target.shape)
    return shapes


def prepare_npy_root():
    required = (
        "train_pos.npy",
        "valid_pos.npy",
        "train_neg.npy",
        "valid_neg.npy",
    )
    if all((NPY_ROOT / name).exists() for name in required):
        return NPY_ROOT, []

    annual_roots = sorted(
        path for path in NPY_ROOT.iterdir()
        if path.is_dir() and all((path / name).exists() for name in required)
    )
    if not annual_roots:
        raise FileNotFoundError(
            "no flat or annual NPY index sets under {}".format(NPY_ROOT)
        )

    COMBINED_NPY_ROOT.mkdir(parents=True, exist_ok=True)
    for name in required:
        combined = []
        for annual_root in annual_roots:
            rows = np.load(annual_root / name, allow_pickle=False)
            if rows.size == 0:
                continue
            if rows.ndim != 2 or rows.shape[1] != 2:
                raise ValueError(
                    "bad shard index shape for {}: {}".format(
                        annual_root / name, rows.shape
                    )
                )
            for stored_path, count in rows:
                shard_path = Path(str(stored_path))
                if not shard_path.is_absolute():
                    shard_path = annual_root / shard_path
                if not shard_path.exists():
                    raise FileNotFoundError(shard_path)
                combined.append((str(shard_path.resolve()), str(count)))
        np.save(
            COMBINED_NPY_ROOT / name,
            np.asarray(combined, dtype=str),
        )
    return COMBINED_NPY_ROOT, [path.name for path in annual_roots]


def main():
    if REQUIREMENTS.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )
    npy_root, annual_inputs = prepare_npy_root()
    requested_years = [
        value for value in os.environ.get("TRAINING_YEARS", "").split(",")
        if value
    ]
    if requested_years and annual_inputs != requested_years:
        raise ValueError(
            "mounted annual inputs {} do not match requested years {}".format(
                annual_inputs, requested_years
            )
        )

    models = [name for name in os.environ.get("ENABLED_MODELS", "SAR").split(",") if name]
    workers = int(os.environ.get("NUM_WORKERS", "10"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "256"))
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "1"))
    compressor = os.environ.get("COMPRESSOR", "lz4")
    log_interval = int(os.environ.get("LOG_INTERVAL", "100000"))
    write_batch_size = int(os.environ.get("WRITE_BATCH_SIZE", "512"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_target_types = {
        "SAR": "frame",
        "FT": "frame",
        "PHN": "sample",
        "RUN": "sample",
    }
    unknown = set(models) - set(model_target_types)
    if unknown:
        raise KeyError("unknown models: {}".format(sorted(unknown)))
    target_priority = {
        "frame": ("SAR", "FT"),
        "sample": ("PHN", "RUN"),
    }
    target_builders = {
        target_type: next(model for model in priority if model in models)
        for target_type, priority in target_priority.items()
        if any(model in models for model in priority)
    }

    timings = {}
    out_path = OUTPUT_ROOT / "shared.zarr"
    for target_type in ("frame", "sample"):
        model = target_builders.get(target_type)
        if model is None:
            continue
        model_root = SOURCE_ROOT / model
        converter = model_root / "preprocess" / "npy2zarr.py"
        if not converter.exists():
            raise FileNotFoundError(converter)
        command = [
            sys.executable,
            converter,
            "--npy_root", npy_root,
            "--out_path", out_path,
            "--num_workers", workers,
            "--chunk_size", chunk_size,
            "--prefetch_factor", prefetch_factor,
            "--compressor", compressor,
            "--log_interval", log_interval,
        ]
        if model != "SAR":
            command.extend(["--write_batch_size", write_batch_size])
        print(
            "building {} targets with {}: {}".format(
                target_type, model, " ".join(map(str, command))
            ),
            flush=True,
        )
        started = time.perf_counter()
        subprocess.check_call([str(item) for item in command], cwd=str(converter.parent))
        timings[target_type] = time.perf_counter() - started

    array_shapes = validate_shared_zarr(out_path, target_builders)
    print(
        "validated positive + negative data and targets in {}".format(out_path),
        flush=True,
    )

    manifest = {
        "sample_run": os.environ.get("SAMPLE_RUN", os.environ.get("CUT_RUN")),
        "requested_years": requested_years,
        "models": models,
        "model_target_types": {
            model: model_target_types[model] for model in models
        },
        "target_builders": target_builders,
        "training_mode": "positive_negative",
        "array_shapes": array_shapes,
        "zarr": "shared.zarr",
        "annual_inputs": annual_inputs,
        "num_workers": workers,
        "chunk_size": chunk_size,
        "compressor": compressor,
        "timing_sec": timings,
    }
    (OUTPUT_ROOT / "npy2zarr_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
