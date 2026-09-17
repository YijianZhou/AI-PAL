#!/usr/bin/env python3
"""Submit one independent GPU Processing Job per enabled picker model."""

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

from processing_job.job_common import prefix_has_objects, upload_tree

# ============================================================================
# USER SETTINGS: edit paths, case identity, AWS resources, and runtime here
# ============================================================================

# Case configs use config_ai_pal_<case>.py and config_<model>_<case>.py.
CASE_CODE = "eg"  # Example case; change consistently for another workflow.

# Installed source package and model registry.
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()  # Installed source package.
local_workflow_dir = Path(__file__).resolve().parent  # Copied case workflow.
shared_config = local_workflow_dir / "config_ai_pal_{}.py".format(CASE_CODE)
pal_src_dir = AI_PAL_ROOT / "PAL_src"
pal_source_files = (
    pal_src_dir / "data_pipeline.py",
    pal_src_dir / "data_pipeline_aws.py",
    pal_src_dir / "data_pipeline_training_aws.py",
    pal_src_dir / "training_zarr_dataset.py",
    pal_src_dir / "training_monitor.py",
    pal_src_dir / "torch_backends.py",
)
enabled_models = ("SAR", "FT", "PHN", "RUN")
models = {
    "SAR": {
        "src": AI_PAL_ROOT / "picker_SAR",
        "config": local_workflow_dir / "config_sar_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g4dn.2xlarge",
    },
    "FT": {
        "src": AI_PAL_ROOT / "picker_FT",
        "config": local_workflow_dir / "config_ft_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g4dn.2xlarge",
    },
    "PHN": {
        "src": AI_PAL_ROOT / "picker_PHN",
        "config": local_workflow_dir / "config_phn_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g4dn.2xlarge",
    },
    "RUN": {
        "src": AI_PAL_ROOT / "picker_RUN",
        "config": local_workflow_dir / "config_run_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g4dn.2xlarge",
    },
}

# Job identity and stage artifacts
region = "us-west-2"
training_years = (2020, 2021, 2022, 2023, 2024, 2025)
RUN_VERSION = "v1"
TRAINING_RUN = "%s-%d-%d-rarity-%s" % (
    CASE_CODE, training_years[0], training_years[-1], RUN_VERSION
)
artifact_prefix = "sagemaker/ai-pal/training/" + TRAINING_RUN
ZARR_ARCHIVE = "%s-rarity-v1" % CASE_CODE
zarr_archive_prefix = "sagemaker/ai-pal/zarr-archives/" + ZARR_ARCHIVE
zarr_root_s3_uri = None  # None uses the stable annual Zarr archive above.
resume_existing_checkpoints = False
allow_existing_output_without_resume = False

# GPU training controls
instance_count = 1
volume_size_gb = 225  # ml.g4dn.2xlarge has fixed 225 GiB local storage.
minimum_checkpoint_free_gb = 50
system_and_runtime_reserve_gb = 40  # Container, filesystem, and staged code.
max_runtime_seconds = 432000
num_workers = 8
prefetch_factor = 2
cpu_threads = 4
framework_version = "2.6.0"
python_version = "py312"


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

def make_input(name, uri, local_path):
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=uri,
            local_path=local_path,
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    )


def prefix_has_checkpoints(s3, bucket_name, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket_name, Prefix=prefix.rstrip("/") + "/"
    ):
        if any(row["Key"].endswith(".ckpt") for row in page.get("Contents", [])):
            return True
    return False


def require_annual_zarr(s3, root_uri, years):
    if not root_uri.startswith("s3://"):
        raise ValueError("expected s3:// Zarr archive URI")
    bucket_name, _, prefix = root_uri[5:].partition("/")
    annual_uris = []
    manifests = {}
    for year in years:
        key = "{}/{}.manifest.json".format(prefix.rstrip("/"), year)
        try:
            body = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise FileNotFoundError(
                    "annual Zarr manifest not found: s3://{}/{}".format(
                        bucket_name, key
                    )
                ) from exc
            raise
        manifest = json.loads(body)
        if int(manifest.get("year", -1)) != int(year):
            raise ValueError("{} manifest records year {}".format(
                year, manifest.get("year")
            ))
        if manifest.get("archive_name") != ZARR_ARCHIVE:
            raise ValueError("{} belongs to Zarr archive {!r}, expected {!r}".format(
                year, manifest.get("archive_name"), ZARR_ARCHIVE
            ))
        manifests[str(year)] = manifest
        annual_uris.append("{}/{}.zarr/".format(root_uri.rstrip("/"), year))
    return annual_uris, manifests


def main():
    if not training_years:
        raise ValueError("training_years must contain at least one year")
    if tuple(training_years) != tuple(range(
        training_years[0], training_years[-1] + 1
    )):
        raise ValueError(
            "training_years must be a chronological contiguous range so "
            "TRAINING_RUN identifies it unambiguously"
        )
    processing_dir = Path(__file__).resolve().parent / "processing_job"
    entry = processing_dir / "processing_entry_train.py"
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
    zarr_root_uri = zarr_root_s3_uri or "s3://{}/{}/".format(
        bucket, zarr_archive_prefix
    )
    annual_zarr_uris, annual_manifests = require_annual_zarr(
        s3, zarr_root_uri, training_years
    )
    annual_sizes = [
        manifest.get("artifact_bytes") for manifest in annual_manifests.values()
    ]
    if all(size is not None for size in annual_sizes):
        input_gib = sum(int(size) for size in annual_sizes) / float(1024 ** 3)
        safe_input_gib = (
            volume_size_gb
            - minimum_checkpoint_free_gb
            - system_and_runtime_reserve_gb
        )
        print("selected annual Zarr input: {:.2f} GiB".format(input_gib))
        if input_gib > safe_input_gib:
            raise RuntimeError(
                "selected annual Zarr stores need {:.2f} GiB, but this instance "
                "allows about {:.2f} GiB after reserving {} GiB for checkpoints "
                "and {} GiB for the system/runtime".format(
                    input_gib, safe_input_gib, minimum_checkpoint_free_gb,
                    system_and_runtime_reserve_gb,
                )
            )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    for model_name in enabled_models:
        model = models[model_name]
        for path in (model["src"], model["config"], model["src"] / model["train_script"]):
            if not path.exists():
                raise FileNotFoundError(path)
        model_code = model_name.lower()
        job_code = "ai-pal-train-{}-{}".format(model_code, TRAINING_RUN)
        job_name = job_code + "-" + timestamp
        stage_prefix = artifact_prefix + "/jobs/" + job_name + "/model"
        output_prefix = artifact_prefix + "/03_checkpoints/" + model_name
        output_uri = "s3://{}/{}/".format(bucket, output_prefix)
        has_output = prefix_has_objects(s3, bucket, output_prefix)
        has_checkpoints = prefix_has_checkpoints(s3, bucket, output_prefix)
        if has_output and not resume_existing_checkpoints and not allow_existing_output_without_resume:
            raise FileExistsError(
                (
                    "checkpoint output exists at {}; enable resume or choose a new "
                    "TRAINING_RUN"
                ).format(output_uri)
            )

        with tempfile.TemporaryDirectory(prefix="ai-pal-train-") as temp_dir:
            model_stage = Path(temp_dir) / "model"
            shutil.copytree(
                model["src"],
                model_stage,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            shutil.copy2(model["config"], model_stage / "config.py")
            shutil.copy2(shared_config, model_stage / "config_ai_pal.py")
            shutil.copy2(requirements, model_stage / "requirements.txt")
            pal_src_stage = model_stage / "PAL_src"
            pal_src_stage.mkdir()
            shutil.copy2(shared_config, pal_src_stage / "config_ai_pal.py")
            for path in pal_source_files:
                shutil.copy2(path, pal_src_stage / path.name)
            upload_tree(s3, model_stage, bucket, stage_prefix)

        source_uri = "s3://{}/{}/".format(bucket, stage_prefix)
        inputs = [
            make_input("model-source", source_uri, "/opt/ml/processing/model"),
            *[
                make_input(
                    "zarr-{}".format(year),
                    year_uri,
                    "/opt/ml/processing/input/zarr/{}.zarr".format(year),
                )
                for year, year_uri in zip(training_years, annual_zarr_uris)
            ],
        ]
        if has_checkpoints and resume_existing_checkpoints:
            inputs.append(make_input("resume-checkpoints", output_uri, "/opt/ml/processing/resume"))
            print("{} resumes artifacts from {}".format(model_name, output_uri))
        elif has_output and resume_existing_checkpoints:
            print("{} found prior status/failure artifacts but no checkpoint to resume".format(
                model_name
            ))

        image_uri = retrieve(
            framework="pytorch",
            region=region,
            version=framework_version,
            py_version=python_version,
            instance_type=model["instance_type"],
            image_scope="training",
        )
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            role=role,
            instance_type=model["instance_type"],
            instance_count=instance_count,
            volume_size_in_gb=volume_size_gb,
            max_runtime_in_seconds=max_runtime_seconds,
            base_job_name=job_code,
            sagemaker_session=session,
            env={
                "MODEL_NAME": model_name,
                "TRAIN_SCRIPT": model["train_script"],
                "TRAINING_MODE": "positive_negative",
                "ZARR_ARCHIVE": ZARR_ARCHIVE,
                "TRAINING_YEARS": ",".join(map(str, training_years)),
                "ZARR_ROOT": "/opt/ml/processing/input/zarr",
                "AI_PAL_TRAINING_YEARS": ",".join(map(str, training_years)),
                "NUM_WORKERS": str(num_workers),
                "PREFETCH_FACTOR": str(prefetch_factor),
                "MIN_CHECKPOINT_FREE_GB": str(minimum_checkpoint_free_gb),
                "PYTHONPATH": "/opt/ml/processing/model/PAL_src",
                "OMP_NUM_THREADS": str(cpu_threads),
                "OPENBLAS_NUM_THREADS": str(cpu_threads),
                "MKL_NUM_THREADS": str(cpu_threads),
            },
        )
        processor.run(
            code=str(entry),
            inputs=inputs,
            outputs=[
                ProcessingOutput(
                    output_name="checkpoints",
                    s3_output=ProcessingS3Output(
                        s3_uri=output_uri,
                        local_path="/opt/ml/processing/output/checkpoints",
                        s3_upload_mode="Continuous",
                    ),
                )
            ],
            job_name=job_name,
            wait=False,
            logs=False,
        )
        print("submitted {}: {}".format(model_name, job_name))
        print("  archive: " + zarr_root_uri)
        print("  years:   " + ", ".join(map(str, training_years)))
        print("  output: " + output_uri)
        print("  volume: {} GiB; require at least {} GiB free for checkpoints".format(
            volume_size_gb, minimum_checkpoint_free_gb
        ))

    print("monitor: python processing_job/monitor_train_jobs.py")


if __name__ == "__main__":
    main()
