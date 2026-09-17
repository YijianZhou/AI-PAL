#!/usr/bin/env python3
"""Submit one CPU job that builds shared data plus required target Zarr arrays."""

import calendar
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)

from processing_job.job_common import (
    delete_s3_prefix,
    prefix_has_objects,
    upload_tree,
)

# ============================================================================
# USER SETTINGS: edit paths, case identity, AWS resources, and runtime here
# ============================================================================

# Case configs use config_ai_pal_<case>.py and config_<model>_<case>.py.
CASE_CODE = "eg"  # Example case; change consistently for another workflow.

# Installed source package and case workflow files.
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()  # Installed source package.
local_workflow_dir = Path(__file__).resolve().parent  # Copied case workflow.
shared_config = local_workflow_dir / "config_ai_pal_{}.py".format(CASE_CODE)
pal_src_dir = AI_PAL_ROOT / "PAL_src"
pal_source_files = (
    pal_src_dir / "data_pipeline.py",
    pal_src_dir / "data_pipeline_aws.py",
    pal_src_dir / "data_pipeline_training_aws.py",
    pal_src_dir / "training_zarr_dataset.py",
)

# Models and their single shared configs
enabled_models = ("SAR", "FT", "PHN", "RUN")
models = {
    "SAR": (AI_PAL_ROOT / "picker_SAR", local_workflow_dir / "config_sar_{}.py".format(CASE_CODE)),
    "FT": (AI_PAL_ROOT / "picker_FT", local_workflow_dir / "config_ft_{}.py".format(CASE_CODE)),
    "PHN": (AI_PAL_ROOT / "picker_PHN", local_workflow_dir / "config_phn_{}.py".format(CASE_CODE)),
    "RUN": (AI_PAL_ROOT / "picker_RUN", local_workflow_dir / "config_run_{}.py".format(CASE_CODE)),
}
model_target_types = {
    "SAR": "frame",
    "FT": "frame",
    "PHN": "sample",
    "RUN": "sample",
}

# Job identity and reusable archive artifacts
region = "us-west-2"
SAMPLE_RUN = "%s-2020-2025-rarity-v1" % CASE_CODE
zarr_years = (2020, 2021, 2022, 2023, 2024, 2025)
ZARR_ARCHIVE = "%s-rarity-v1" % CASE_CODE
job_code = "ai-pal-zarr-" + ZARR_ARCHIVE
artifact_prefix = "sagemaker/ai-pal/zarr-archives/" + ZARR_ARCHIVE
sample_artifact_prefix = "sagemaker/ai-pal/training/" + SAMPLE_RUN
npy_s3_root_uri = None  # None uses <default bucket>/<sample run>/01_npy/.
overwrite_existing_years = False

# CPU conversion controls
instance_type = "ml.c5.4xlarge"
instance_count = 1
volume_size_gb = 1024
max_runtime_seconds = 432000
num_workers = 8
chunk_size = 256
prefetch_factor = 1
compressor = "lz4"
write_batch_size = 512
log_interval = 100000
threads_per_worker = 1
framework_version = "2.6.0"
python_version = "py312"


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================


def split_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError("expected s3:// URI, got {}".format(uri))
    bucket_name, _, key = uri[5:].partition("/")
    if not bucket_name:
        raise ValueError("S3 URI has no bucket: {}".format(uri))
    return bucket_name, key.rstrip("/")


def require_complete_annual_npy(s3, root_uri, years):
    root_bucket, root_key = split_s3_uri(root_uri)
    required = (
        "train_pos.npy",
        "valid_pos.npy",
        "train_neg.npy",
        "valid_neg.npy",
        "cut_samples_manifest.json",
    )
    year_uris = []
    for year in years:
        year_key = "{}/{}".format(root_key, year)
        missing = []
        for name in required:
            key = "{}/{}".format(year_key, name)
            try:
                s3.head_object(Bucket=root_bucket, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey", "NotFound"):
                    missing.append(name)
                else:
                    raise
        if missing:
            raise FileNotFoundError(
                "{} NPY output is incomplete under s3://{}/{}; missing {}".format(
                    year, root_bucket, year_key, ", ".join(missing)
                )
            )
        manifest_key = "{}/cut_samples_manifest.json".format(year_key)
        manifest = json.loads(
            s3.get_object(Bucket=root_bucket, Key=manifest_key)["Body"].read()
        )
        if int(manifest.get("training_year", -1)) != int(year):
            raise ValueError(
                "{} manifest training_year is {!r}".format(
                    year, manifest.get("training_year")
                )
            )
        phase_file = str(manifest.get("phase_file", ""))
        if not phase_file.endswith("_pal_rarity.pha"):
            raise ValueError(
                "{} manifest is not rarity-aware: {!r}".format(year, phase_file)
            )
        expected_dates = 366 if calendar.isleap(int(year)) else 365
        if int(manifest.get("association_rate_dates", -1)) != expected_dates:
            raise ValueError(
                "{} manifest has {} association-rate dates; expected {}".format(
                    year,
                    manifest.get("association_rate_dates"),
                    expected_dates,
                )
            )
        year_uris.append(
            "s3://{}/{}/".format(root_bucket, year_key)
        )
    return year_uris


def require_positive_negative_converter(converter, target_type):
    """Reject stale converters before paying for a Processing Job."""
    source = converter.read_text(encoding="utf-8")
    required_tokens = (
        "train_pos.npy",
        "valid_pos.npy",
        "train_neg.npy",
        "valid_neg.npy",
        "_target_{}".format(target_type),
    )
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise RuntimeError(
            "{} does not satisfy the positive/negative Zarr contract; missing {}. "
            "Update AI_PAL_ROOT before submitting.".format(
                converter, ", ".join(missing)
            )
        )


def main():
    if not zarr_years:
        raise ValueError("zarr_years must contain at least one year")
    if len(set(zarr_years)) != len(zarr_years):
        raise ValueError("zarr_years contains duplicates")
    if tuple(sorted(zarr_years)) != tuple(zarr_years):
        raise ValueError("zarr_years must be in chronological order")

    processing_dir = Path(__file__).resolve().parent / "processing_job"
    entry = processing_dir / "processing_entry_npy2zarr.py"
    requirements = processing_dir / "requirements.txt"
    for path in (entry, requirements, shared_config, *pal_source_files):
        if not path.exists():
            raise FileNotFoundError(path)
    unknown = set(enabled_models) - set(models)
    if unknown:
        raise KeyError("unknown models: {}".format(sorted(unknown)))

    session = Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    bucket = session.default_bucket()
    s3 = boto3.client("s3", region_name=region)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    year_label = "{}-{}".format(zarr_years[0], zarr_years[-1])
    job_name = "{}-{}-{}".format(job_code, year_label, timestamp)
    stage_prefix = artifact_prefix + "/jobs/" + job_name + "/source"
    input_root_uri = npy_s3_root_uri or "s3://{}/{}/01_npy/".format(
        bucket, sample_artifact_prefix
    )
    annual_input_uris = require_complete_annual_npy(
        s3, input_root_uri, zarr_years
    )
    output_prefix = artifact_prefix
    output_uri = "s3://{}/{}/".format(bucket, output_prefix)
    for year in zarr_years:
        year_prefix = "{}/{}.zarr".format(output_prefix, year)
        manifest_key = "{}/{}.manifest.json".format(output_prefix, year)
        year_exists = prefix_has_objects(s3, bucket, year_prefix)
        manifest_exists = prefix_has_objects(s3, bucket, manifest_key)
        if year_exists or manifest_exists:
            if not overwrite_existing_years:
                raise FileExistsError(
                    "annual Zarr {} already exists under {}; remove {} from "
                    "zarr_years or set overwrite_existing_years=True".format(
                        year, output_uri, year
                    )
                )
            deleted = delete_s3_prefix(s3, bucket, year_prefix)
            deleted += delete_s3_prefix(s3, bucket, manifest_key)
            print("deleted {} old {} artifact(s)".format(deleted, year))

    with tempfile.TemporaryDirectory(prefix="ai-pal-zarr-") as temp_dir:
        source_stage = Path(temp_dir) / "source"
        source_stage.mkdir()
        shutil.copy2(requirements, source_stage / "requirements.txt")
        pal_src_stage = source_stage / "PAL_src"
        pal_src_stage.mkdir()
        shutil.copy2(shared_config, pal_src_stage / "config_ai_pal.py")
        for path in pal_source_files:
            shutil.copy2(path, pal_src_stage / path.name)
        for model_name in enabled_models:
            model_dir, config_path = models[model_name]
            preprocess_dir = model_dir / "preprocess"
            for path in (preprocess_dir / "npy2zarr.py", config_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            require_positive_negative_converter(
                preprocess_dir / "npy2zarr.py",
                model_target_types[model_name],
            )
            target = source_stage / model_name
            shutil.copytree(
                preprocess_dir,
                target / "preprocess",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            shutil.copy2(config_path, target / "config.py")
            shutil.copy2(shared_config, target / "config_ai_pal.py")
        upload_tree(s3, source_stage, bucket, stage_prefix)

    source_uri = "s3://{}/{}/".format(bucket, stage_prefix)
    image_uri = retrieve(
        framework="pytorch",
        region=region,
        version=framework_version,
        py_version=python_version,
        instance_type=instance_type,
        image_scope="training",
    )
    processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size_in_gb=volume_size_gb,
        max_runtime_in_seconds=max_runtime_seconds,
        base_job_name=job_code,
        sagemaker_session=session,
        env={
            "SAMPLE_RUN": SAMPLE_RUN,
            "ZARR_ARCHIVE": ZARR_ARCHIVE,
            "ZARR_YEARS": ",".join(map(str, zarr_years)),
            "ENABLED_MODELS": ",".join(enabled_models),
            "NUM_WORKERS": str(num_workers),
            "CHUNK_SIZE": str(chunk_size),
            "PREFETCH_FACTOR": str(prefetch_factor),
            "COMPRESSOR": compressor,
            "WRITE_BATCH_SIZE": str(write_batch_size),
            "LOG_INTERVAL": str(log_interval),
            "OUTPUT_S3_URI": output_uri,
            "PYTHONPATH": "/opt/ml/processing/source/PAL_src",
            "OMP_NUM_THREADS": str(threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(threads_per_worker),
            "MKL_NUM_THREADS": str(threads_per_worker),
        },
    )
    processor.run(
        code=str(entry),
        inputs=[
            ProcessingInput(
                input_name="source",
                s3_input=ProcessingS3Input(
                    s3_uri=source_uri,
                    local_path="/opt/ml/processing/source",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                ),
            ),
            *[
                ProcessingInput(
                    input_name="npy-{}".format(year),
                    s3_input=ProcessingS3Input(
                        s3_uri=year_uri,
                        local_path="/opt/ml/processing/input/npy/{}".format(
                            year
                        ),
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                )
                for year, year_uri in zip(
                    zarr_years, annual_input_uris
                )
            ],
        ],
        outputs=[
            ProcessingOutput(
                output_name="zarr-workspace",
                s3_output=ProcessingS3Output(
                    s3_uri=output_uri,
                    local_path="/opt/ml/processing/output/zarr",
                    s3_upload_mode="EndOfJob",
                ),
            )
        ],
        job_name=job_name,
        wait=False,
        logs=False,
    )
    print("submitted: " + job_name)
    print("sample run: " + SAMPLE_RUN)
    print("archive:   " + ZARR_ARCHIVE)
    print("years:     " + ", ".join(map(str, zarr_years)))
    print("volume:    {} GiB (mounted Zarr workspace)".format(
        volume_size_gb
    ))
    for year_uri in annual_input_uris:
        print("input:     " + year_uri)
    print("output:    " + output_uri)
    print("monitor:   python processing_job/monitor_npy2zarr_job.py")


if __name__ == "__main__":
    main()
