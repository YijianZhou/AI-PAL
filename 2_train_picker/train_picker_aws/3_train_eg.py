#!/usr/bin/env python3
"""Submit one independent GPU Processing Job per enabled picker model."""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
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
pal_data_pipeline = AI_PAL_ROOT / "PAL_src" / "data_pipeline.py"
enabled_models = ("SAR", "FT", "PHN", "RUN")
models = {
    "SAR": {
        "src": AI_PAL_ROOT / "picker_SAR",
        "config": local_workflow_dir / "config_sar_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g5.2xlarge",
    },
    "FT": {
        "src": AI_PAL_ROOT / "picker_FT",
        "config": local_workflow_dir / "config_ft_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g5.2xlarge",
    },
    "PHN": {
        "src": AI_PAL_ROOT / "picker_PHN",
        "config": local_workflow_dir / "config_phn_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g5.2xlarge",
    },
    "RUN": {
        "src": AI_PAL_ROOT / "picker_RUN",
        "config": local_workflow_dir / "config_run_{}.py".format(CASE_CODE),
        "train_script": "train.py",
        "instance_type": "ml.g5.2xlarge",
    },
}

# Job identity and stage artifacts
region = "us-west-2"
TRAINING_RUN = "%s-2020-2025-v1" % CASE_CODE
artifact_prefix = "sagemaker/ai-pal/training/" + TRAINING_RUN
zarr_root_s3_uri = None  # None uses <default bucket>/<artifact_prefix>/02_zarr/
resume_existing_checkpoints = True
allow_existing_output_without_resume = False

# GPU training controls
instance_count = 1
volume_size_gb = 1024
max_runtime_seconds = 432000
num_workers = 10
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


def main():
    processing_dir = Path(__file__).resolve().parent / "processing_job"
    entry = processing_dir / "processing_entry_train.py"
    requirements = processing_dir / "requirements.txt"
    for path in (entry, requirements, shared_config, pal_data_pipeline):
        if not path.exists():
            raise FileNotFoundError(path)
    unknown = set(enabled_models) - set(models)
    if unknown:
        raise KeyError("unknown models: {}".format(sorted(unknown)))

    session = Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    bucket = session.default_bucket()
    s3 = boto3.client("s3", region_name=region)
    zarr_root_uri = zarr_root_s3_uri or "s3://{}/{}/02_zarr/".format(
        bucket, artifact_prefix
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
            shutil.copy2(requirements, model_stage / "requirements.txt")
            pal_src_stage = model_stage / "PAL_src"
            pal_src_stage.mkdir()
            shutil.copy2(shared_config, pal_src_stage / "config_ai_pal.py")
            shutil.copy2(pal_data_pipeline, pal_src_stage / "data_pipeline.py")
            upload_tree(s3, model_stage, bucket, stage_prefix)

        source_uri = "s3://{}/{}/".format(bucket, stage_prefix)
        zarr_uri = zarr_root_uri.rstrip("/") + "/shared.zarr/"
        inputs = [
            make_input("model-source", source_uri, "/opt/ml/processing/model"),
            make_input(
                "zarr-dataset",
                zarr_uri,
                "/opt/ml/processing/input/zarr/shared.zarr",
            ),
        ]
        if has_output and resume_existing_checkpoints:
            inputs.append(make_input("resume-checkpoints", output_uri, "/opt/ml/processing/resume"))
            print("{} resumes artifacts from {}".format(model_name, output_uri))

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
                "ZARR_NAME": "shared.zarr",
                "NUM_WORKERS": str(num_workers),
                "PREFETCH_FACTOR": str(prefetch_factor),
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
        print("  input:  " + zarr_uri)
        print("  output: " + output_uri)

    print("monitor: python processing_job/monitor_train_jobs.py")


if __name__ == "__main__":
    main()
