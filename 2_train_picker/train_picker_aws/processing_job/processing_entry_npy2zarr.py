#!/usr/bin/env python3
"""Build and publish independently reusable annual AI-PAL Zarr stores."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


SOURCE_ROOT = Path("/opt/ml/processing/source")
NPY_ROOT = Path("/opt/ml/processing/input/npy")
OUTPUT_ROOT = Path("/opt/ml/processing/output/zarr")
REQUIREMENTS = SOURCE_ROOT / "requirements.txt"
UPLOAD_WORKERS = 16


def split_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError("expected s3:// URI, got {}".format(uri))
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError("S3 URI has no bucket: {}".format(uri))
    return bucket, prefix.rstrip("/")


def upload_file(path, output_uri, relative):
    bucket, prefix = split_s3_uri(output_uri)
    key = "/".join(filter(None, (prefix, relative)))
    boto3.client("s3").upload_file(str(path), bucket, key)


def resource_snapshot():
    lines = []
    for label, path in (
        ("input", NPY_ROOT),
        ("output", OUTPUT_ROOT.parent),
        ("shared_memory", Path("/dev/shm")),
    ):
        if path.exists():
            usage = shutil.disk_usage(path)
            lines.append(
                "{} disk: total={:.2f} GiB used={:.2f} GiB free={:.2f} GiB".format(
                    label,
                    usage.total / 1024 ** 3,
                    usage.used / 1024 ** 3,
                    usage.free / 1024 ** 3,
                )
            )
    return lines


def remove_local_zarrs():
    for path in OUTPUT_ROOT.glob("*.zarr"):
        if path.is_dir():
            print("removing local Zarr workspace {}".format(path), flush=True)
            shutil.rmtree(path)


def run_logged_converter(command, cwd, log_path, output_uri, log_key):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = []
    last_upload = 0.0
    with log_path.open("w", encoding="utf-8") as log_fp:
        for line in resource_snapshot():
            print(line, flush=True)
            log_fp.write(line + "\n")
        log_fp.flush()
        upload_file(log_path, output_uri, log_key)
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_fp.write(line)
            log_fp.flush()
            tail.append(line.rstrip("\n"))
            tail = tail[-120:]
            now = time.monotonic()
            if now - last_upload >= 60.0:
                log_fp.write("--- resources {} ---\n".format(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ))
                for resource_line in resource_snapshot():
                    log_fp.write(resource_line + "\n")
                log_fp.flush()
                upload_file(log_path, output_uri, log_key)
                last_upload = now
        return_code = process.wait()
    upload_file(log_path, output_uri, log_key)
    if return_code:
        raise RuntimeError(
            "converter exited with status {}\n--- converter log tail ---\n{}".format(
                return_code, "\n".join(tail)
            )
        )


def upload_output_tree(root, output_uri, previous_state=None):
    bucket, prefix = split_s3_uri(output_uri)
    previous_state = previous_state or {}
    current_state = {}
    pending = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        current_state[relative] = signature
        if previous_state.get(relative) != signature:
            pending.append((path, relative, stat.st_size))
    if not pending:
        return current_state

    client = boto3.client(
        "s3",
        config=BotoConfig(
            max_pool_connections=UPLOAD_WORKERS,
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )
    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
    )

    def upload_one(item):
        path, relative, size = item
        key = "/".join(part for part in (prefix, relative) if part)
        client.upload_file(str(path), bucket, key, Config=transfer_config)
        return size

    uploaded_files = 0
    uploaded_bytes = 0
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
        futures = [executor.submit(upload_one, item) for item in pending]
        for future in as_completed(futures):
            uploaded_bytes += future.result()
            uploaded_files += 1
            if uploaded_files == len(pending) or uploaded_files % 1000 == 0:
                print(
                    "uploaded {:,}/{:,} files ({:.2f} GiB)".format(
                        uploaded_files,
                        len(pending),
                        uploaded_bytes / float(1024 ** 3),
                    ),
                    flush=True,
                )
    return current_state


def validate_annual_zarr(zarr_path, target_builders):
    import zarr

    shapes = {}
    metadata = {}
    for split in ("train", "valid"):
        for sample_kind in ("positive", "negative"):
            data_name = "{}/{}_data".format(split, sample_kind)
            data_path = zarr_path / data_name
            if not data_path.exists():
                raise FileNotFoundError("missing required Zarr array: " + data_name)
            data = zarr.open(str(data_path), mode="r")
            if data.shape[0] == 0:
                raise ValueError("empty required Zarr array: " + data_name)
            shapes[data_name] = list(data.shape)
            metadata[data_name] = {
                "chunks": list(data.chunks),
                "dtype": str(data.dtype),
            }
            for target_type in target_builders:
                target_name = "{}/{}_target_{}".format(
                    split, sample_kind, target_type
                )
                target_path = zarr_path / target_name
                if not target_path.exists():
                    raise FileNotFoundError(
                        "missing required Zarr array: " + target_name
                    )
                target = zarr.open(str(target_path), mode="r")
                if target.shape[0] != data.shape[0]:
                    raise ValueError(
                        "sample-count mismatch: {} {} vs {} {}".format(
                            data_name, data.shape, target_name, target.shape
                        )
                    )
                shapes[target_name] = list(target.shape)
                metadata[target_name] = {
                    "chunks": list(target.chunks),
                    "dtype": str(target.dtype),
                }
    return shapes, metadata


def mounted_annual_roots(requested_years):
    required = ("train_pos.npy", "valid_pos.npy", "train_neg.npy", "valid_neg.npy")
    roots = {}
    for year in requested_years:
        root = NPY_ROOT / str(year)
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(
                "{} annual NPY input is missing {}".format(year, ", ".join(missing))
            )
        roots[str(year)] = root
    return roots


def read_archive_manifest(output_uri):
    bucket, prefix = split_s3_uri(output_uri)
    key = "/".join(filter(None, (prefix, "archive_manifest.json")))
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def validate_archive_compatibility(archive, year, shapes):
    if archive.get("archive_name") != os.environ.get("ZARR_ARCHIVE"):
        raise ValueError(
            "existing archive name {!r} does not match {!r}".format(
                archive.get("archive_name"), os.environ.get("ZARR_ARCHIVE")
            )
        )
    for existing_year, row in sorted(archive.get("years", {}).items()):
        existing_shapes = row.get("array_shapes", {})
        if not existing_shapes:
            continue
        if set(existing_shapes) != set(shapes):
            raise ValueError(
                "{} arrays differ from archived year {}".format(year, existing_year)
            )
        for name in shapes:
            if list(existing_shapes[name][1:]) != list(shapes[name][1:]):
                raise ValueError(
                    "{} {} shape {} is incompatible with {} shape {}".format(
                        year, name, shapes[name], existing_year,
                        existing_shapes[name]
                    )
                )
        return


def main():
    if REQUIREMENTS.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )
    requested_years = [
        value for value in os.environ.get("ZARR_YEARS", "").split(",") if value
    ]
    if not requested_years:
        raise ValueError("ZARR_YEARS is required")
    annual_roots = mounted_annual_roots(requested_years)
    models = [
        name for name in os.environ.get("ENABLED_MODELS", "SAR").split(",") if name
    ]
    workers = int(os.environ.get("NUM_WORKERS", "10"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "256"))
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "1"))
    compressor = os.environ.get("COMPRESSOR", "lz4")
    log_interval = int(os.environ.get("LOG_INTERVAL", "100000"))
    write_batch_size = int(os.environ.get("WRITE_BATCH_SIZE", "512"))
    output_s3_uri = os.environ["OUTPUT_S3_URI"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model_target_types = {
        "SAR": "frame", "FT": "frame", "PHN": "sample", "RUN": "sample",
    }
    unknown = set(models) - set(model_target_types)
    if unknown:
        raise KeyError("unknown models: {}".format(sorted(unknown)))
    target_priority = {"frame": ("SAR", "FT"), "sample": ("PHN", "RUN")}
    target_builders = {
        target_type: next(model for model in priority if model in models)
        for target_type, priority in target_priority.items()
        if any(model in models for model in priority)
    }

    archive = read_archive_manifest(output_s3_uri) or {
        "schema_version": 1,
        "archive_name": os.environ.get("ZARR_ARCHIVE"),
        "years": {},
    }
    upload_state = {}
    for year in requested_years:
        npy_root = annual_roots[year]
        out_path = OUTPUT_ROOT / "{}.zarr".format(year)
        timings = {}
        for target_type in ("frame", "sample"):
            model = target_builders.get(target_type)
            if model is None:
                continue
            converter = SOURCE_ROOT / model / "preprocess" / "npy2zarr.py"
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
            log_key = "logs/{}_{}_{}.log".format(year, target_type, model)
            started = time.perf_counter()
            run_logged_converter(
                command,
                converter.parent,
                OUTPUT_ROOT / log_key,
                output_s3_uri,
                log_key,
            )
            timings[target_type] = time.perf_counter() - started

        shapes, array_metadata = validate_annual_zarr(out_path, target_builders)
        validate_archive_compatibility(archive, year, shapes)
        artifact_files = sum(1 for path in out_path.rglob("*") if path.is_file())
        artifact_bytes = sum(
            path.stat().st_size for path in out_path.rglob("*") if path.is_file()
        )
        year_manifest = {
            "schema_version": 1,
            "year": int(year),
            "sample_run": os.environ.get("SAMPLE_RUN"),
            "archive_name": os.environ.get("ZARR_ARCHIVE"),
            "models": models,
            "model_target_types": {
                model: model_target_types[model] for model in models
            },
            "target_builders": target_builders,
            "training_mode": "positive_negative",
            "array_shapes": shapes,
            "array_metadata": array_metadata,
            "zarr": "{}.zarr".format(year),
            "num_workers": workers,
            "chunk_size": chunk_size,
            "compressor": compressor,
            "timing_sec": timings,
            "artifact_files": artifact_files,
            "artifact_bytes": artifact_bytes,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = OUTPUT_ROOT / "{}.manifest.json".format(year)
        manifest_path.write_text(
            json.dumps(year_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        upload_state = upload_output_tree(OUTPUT_ROOT, output_s3_uri, upload_state)
        archive["years"][year] = {
            "manifest": "{}.manifest.json".format(year),
            "zarr": "{}.zarr".format(year),
            "sample_run": year_manifest["sample_run"],
            "array_shapes": shapes,
            "artifact_files": artifact_files,
            "artifact_bytes": artifact_bytes,
            "completed_utc": year_manifest["completed_utc"],
        }
        archive["updated_utc"] = datetime.now(timezone.utc).isoformat()
        archive_path = OUTPUT_ROOT / "archive_manifest.json"
        archive_path.write_text(
            json.dumps(archive, indent=2, sort_keys=True), encoding="utf-8"
        )
        upload_state = upload_output_tree(OUTPUT_ROOT, output_s3_uri, upload_state)
        print("published annual Zarr {}".format(year), flush=True)
        shutil.rmtree(out_path)

    print(json.dumps(archive, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure_text = traceback.format_exc()
        print(failure_text, flush=True)
        failure_uri = os.environ.get("OUTPUT_S3_URI")
        if failure_uri:
            bucket, prefix = split_s3_uri(failure_uri)
            key = "/".join(filter(None, (prefix, "npy2zarr_failure.txt")))
            boto3.client("s3").put_object(
                Bucket=bucket,
                Key=key,
                Body=failure_text.encode("utf-8"),
                ContentType="text/plain",
            )
        remove_local_zarrs()
        raise
