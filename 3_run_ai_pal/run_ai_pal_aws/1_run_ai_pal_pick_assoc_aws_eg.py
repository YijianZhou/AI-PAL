#!/usr/bin/env python3
"""Submit one resumable AWS job for AI-PAL picking and association."""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input

from processing_job.job_common import prefix_has_objects, upload_tree


# ============================================================================
# USER SETTINGS: case, dates, station sets, trained models, and AWS resources
# ============================================================================
CASE_CODE = "eg"
TIME_RANGE = "20190704-20190707"  # Target dates; exclusive end date.
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()
WORKFLOW_DIR = Path(__file__).resolve().parent
FULL_STATION_FILE = Path("input/example_pal_format1.sta")
SUBNET_STATION_FILES = []

# POS_NEG checkpoints are the outputs from 2_train_picker.
TRAINING_RUN = "eg-2020-2025-rarity-v1"
CHECKPOINT_S3_ROOT_URI = None
INFERENCE_RUN = "eg-ai-pal-20190704-20190707-v1"

# The standard four-GPU mapping mirrors the local workflow. For a one-GPU
# instance, map every model to 0 and verify that all selected models fit VRAM.
MODEL_GPU_MAP = {"SAR": 0, "FT": 1, "PHN": 2, "RUN": 3}
instance_type = "ml.g4dn.12xlarge"
instance_count = 1
volume_size_gb = 512
max_runtime_seconds = 432000
num_pick_workers = 12
num_assoc_workers = 12
cpu_threads = 1
overwrite_picks = False
resume_existing_output = True

region = "us-west-2"
framework_version = "2.6.0"
python_version = "py312"


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================
MODEL_NAMES = ("SAR", "FT", "PHN", "RUN")
POSITIVE_CHECKPOINTS = (
    "ceed_pos_sar_best.ckpt", "ceed_pos_ft_best.ckpt",
    "ceed_pos_phn_best.ckpt", "ceed_pos_run_best.ckpt",
)
PAL_SOURCE_NAMES = (
    "associator_pal.py", "association_runner.py", "data_pipeline.py",
    "data_pipeline_aws.py", "data_pipeline_ai_aws.py", "event_repicker.py",
    "offline_pick_assoc_runner.py", "offline_picker_runner.py",
    "rolling_waveform.py",
    "phase_merge.py", "pick_ensemble.py", "picker_stream.py",
    "runtime_console.py", "torch_backends.py", "trigger_counts.py",
    "station_sets.py", "waveform_qc.py",
)


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


def split_s3_uri(uri):
    bucket, _, prefix = uri[5:].partition("/")
    if not uri.startswith("s3://") or not bucket:
        raise ValueError("invalid S3 URI: {}".format(uri))
    return bucket, prefix.rstrip("/")


def main():
    processing_dir = WORKFLOW_DIR / "processing_job"
    entry = processing_dir / "processing_entry_ai_pal.py"
    requirements = processing_dir / "requirements.txt"
    shared_config = WORKFLOW_DIR / "config_ai_pal_{}.py".format(CASE_CODE)
    station_paths = [FULL_STATION_FILE, *SUBNET_STATION_FILES]
    station_paths = [
        path if path.is_absolute() else WORKFLOW_DIR / path
        for path in station_paths
    ]
    model_dirs = {
        name: AI_PAL_ROOT / "picker_{}".format(name)
        for name in MODEL_NAMES
    }
    required = [entry, requirements, shared_config, *station_paths]
    required.extend(AI_PAL_ROOT / "PAL_src" / name for name in PAL_SOURCE_NAMES)
    for model in MODEL_NAMES:
        required.extend((
            model_dirs[model],
            WORKFLOW_DIR / "config_{}_{}.py".format(model.lower(), CASE_CODE),
            WORKFLOW_DIR / "config_{}_pos_ceed.py".format(model.lower()),
        ))
    required.extend(
        WORKFLOW_DIR / "input" / "CEED_ckpt" / name
        for name in POSITIVE_CHECKPOINTS
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    session = Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    bucket = session.default_bucket()
    s3 = boto3.client("s3", region_name=region)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_code = "ai-pal-run-" + INFERENCE_RUN
    job_name = job_code + "-" + timestamp
    artifact_prefix = "sagemaker/ai-pal/inference/" + INFERENCE_RUN
    stage_prefix = artifact_prefix + "/jobs/" + job_name + "/source"
    output_prefix = artifact_prefix + "/output"
    output_uri = "s3://{}/{}/".format(bucket, output_prefix)
    checkpoint_root_uri = CHECKPOINT_S3_ROOT_URI or (
        "s3://{}/sagemaker/ai-pal/training/{}/03_checkpoints/".format(
            bucket, TRAINING_RUN
        )
    )
    checkpoint_bucket, checkpoint_prefix = split_s3_uri(checkpoint_root_uri)

    inputs = []
    with tempfile.TemporaryDirectory(prefix="ai-pal-run-") as temp_dir:
        stage = Path(temp_dir) / "source"
        pal_stage = stage / "PAL_src"
        workflow_stage = stage / "workflow"
        input_stage = workflow_stage / "input"
        ceed_stage = input_stage / "CEED_ckpt"
        pal_stage.mkdir(parents=True)
        ceed_stage.mkdir(parents=True)
        shutil.copy2(requirements, stage / "requirements.txt")
        shutil.copy2(shared_config, pal_stage / "config_ai_pal.py")
        for name in PAL_SOURCE_NAMES:
            shutil.copy2(AI_PAL_ROOT / "PAL_src" / name, pal_stage / name)
        for model in MODEL_NAMES:
            shutil.copy2(
                WORKFLOW_DIR / "config_{}_{}.py".format(model.lower(), CASE_CODE),
                workflow_stage / "config_{}_case.py".format(model.lower()),
            )
            shutil.copy2(
                WORKFLOW_DIR / "config_{}_pos_ceed.py".format(model.lower()),
                workflow_stage / "config_{}_pos_ceed.py".format(model.lower()),
            )
        for path in station_paths:
            shutil.copy2(path, input_stage / path.name)
        for path in (WORKFLOW_DIR / "input" / "CEED_ckpt").glob("*.ckpt"):
            shutil.copy2(path, ceed_stage / path.name)
        upload_tree(s3, stage, bucket, stage_prefix)
        for model, model_dir in model_dirs.items():
            upload_tree(
                s3, model_dir, bucket,
                stage_prefix + "/picker_{}".format(model),
            )

    source_uri = "s3://{}/{}/".format(bucket, stage_prefix)
    inputs.append(make_input("source", source_uri, "/opt/ml/processing/source"))
    for model in MODEL_NAMES:
        model_prefix = checkpoint_prefix + "/" + model
        if not prefix_has_objects(s3, checkpoint_bucket, model_prefix):
            raise FileNotFoundError(
                "no {} checkpoint objects under s3://{}/{}".format(
                    model, checkpoint_bucket, model_prefix
                )
            )
        inputs.append(make_input(
            "checkpoint-" + model.lower(),
            "s3://{}/{}/".format(checkpoint_bucket, model_prefix),
            "/opt/ml/processing/checkpoints/" + model,
        ))
    if prefix_has_objects(s3, bucket, output_prefix):
        if not resume_existing_output:
            raise FileExistsError(
                "output already exists at {}; enable resume or change INFERENCE_RUN"
                .format(output_uri)
            )
        inputs.append(make_input(
            "resume", output_uri, "/opt/ml/processing/resume"
        ))
        print("resuming from: {}".format(output_uri))

    image_uri = retrieve(
        framework="pytorch", region=region, version=framework_version,
        py_version=python_version, instance_type=instance_type,
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
            "CASE_CODE": CASE_CODE,
            "TIME_RANGE": TIME_RANGE,
            "FULL_STATION_FILE": station_paths[0].name,
            "SUBNET_STATION_FILES": json.dumps({
                "r{}".format(index): path.name
                for index, path in enumerate(station_paths[1:], start=1)
            }),
            "MODEL_GPU_MAP": json.dumps(MODEL_GPU_MAP),
            "NUM_PICK_WORKERS": str(num_pick_workers),
            "NUM_ASSOC_WORKERS": str(num_assoc_workers),
            "OVERWRITE_PICKS": "1" if overwrite_picks else "0",
            "OUTPUT_S3_URI": output_uri,
            "SCEDC_ACCESS_MODE": "signed",
            "SCEDC_REGION": region,
            "OMP_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "PYTHONPATH": "/opt/ml/processing/source:/opt/ml/processing/source/PAL_src",
        },
    )
    processor.run(
        code=str(entry), inputs=inputs, outputs=[], job_name=job_name,
        wait=False, logs=False,
    )
    print("submitted: {}".format(job_name))
    print("output:    {}".format(output_uri))
    print("monitor:   python processing_job/monitor_ai_pal_job.py")


if __name__ == "__main__":
    main()
