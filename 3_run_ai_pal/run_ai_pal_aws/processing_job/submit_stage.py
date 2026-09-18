"""Shared SageMaker submission mechanics for staged AI-PAL inference."""

import shutil
import tempfile
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingS3Input

from processing_job.job_common import prefix_has_objects, upload_tree


MODEL_NAMES = ("SAR", "FT", "PHN", "RUN")
POSITIVE_CHECKPOINTS = (
    "ceed_pos_sar_best.ckpt", "ceed_pos_ft_best.ckpt",
    "ceed_pos_phn_best.ckpt", "ceed_pos_run_best.ckpt",
)
PAL_SOURCE_NAMES = (
    "associator_pal.py", "association_runner.py", "aws_inference_job.py",
    "data_pipeline.py", "data_pipeline_aws.py", "data_pipeline_ai_aws.py",
    "event_repicker.py", "offline_event_postprocessor.py",
    "offline_pick_assoc_runner.py", "offline_picker_runner.py",
    "rolling_waveform.py",
    "phase_merge.py", "pick_ensemble.py", "picker_stream.py",
    "runtime_console.py", "torch_backends.py", "trigger_counts.py",
    "station_sets.py", "waveform_qc.py",
)


def _input(name, uri, local_path):
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=uri, local_path=local_path,
            s3_data_type="S3Prefix", s3_input_mode="File",
        ),
    )


def _split_s3_uri(uri):
    bucket, _, prefix = uri[5:].partition("/")
    if not uri.startswith("s3://") or not bucket:
        raise ValueError("invalid S3 URI: {}".format(uri))
    return bucket, prefix.rstrip("/")


def _require_object(s3, bucket, key, label):
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "404", "NoSuchKey", "NotFound"
        ):
            raise FileNotFoundError(
                "{} is incomplete; missing s3://{}/{}".format(
                    label, bucket, key
                )
            ) from exc
        raise


def submit_stage(
    *, workflow_dir, ai_pal_root, case_code, time_range, inference_run,
    training_run, checkpoint_s3_root_uri, full_station_file,
    subnet_station_files, stage_code, entry_name, required_prior_manifest,
    include_models, instance_type, instance_count, volume_size_gb,
    max_runtime_seconds, num_workers, model_gpu_map, overwrite,
    extra_env=None, region="us-west-2", framework_version="2.6.0",
    python_version="py312", cpu_threads=1, sync_interval_sec=300,
    include_positive_models=True,
):
    workflow_dir = Path(workflow_dir).resolve()
    ai_pal_root = Path(ai_pal_root).expanduser().resolve()
    processing_dir = workflow_dir / "processing_job"
    entry = processing_dir / entry_name
    requirements = processing_dir / "requirements.txt"
    shared_config = workflow_dir / "config_ai_pal_{}.py".format(case_code)
    station_paths = [full_station_file, *subnet_station_files]
    station_paths = [
        path if Path(path).is_absolute() else workflow_dir / path
        for path in map(Path, station_paths)
    ]
    required = [entry, requirements, shared_config, *station_paths]
    required.extend(ai_pal_root / "PAL_src" / name for name in PAL_SOURCE_NAMES)
    if include_models:
        for model in MODEL_NAMES:
            required.extend((
                ai_pal_root / "picker_{}".format(model),
                workflow_dir / "config_{}_{}.py".format(
                    model.lower(), case_code
                ),
            ))
            if include_positive_models:
                required.append(workflow_dir / "config_{}_pos_ceed.py".format(
                    model.lower()))
        if include_positive_models:
            required.extend(
                workflow_dir / "input" / "CEED_ckpt" / name
                for name in POSITIVE_CHECKPOINTS
            )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    session = Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    bucket = session.default_bucket()
    s3 = boto3.client("s3", region_name=region)
    output_prefix = "sagemaker/ai-pal/inference/{}/output".format(inference_run)
    output_uri = "s3://{}/{}/".format(bucket, output_prefix)
    if required_prior_manifest:
        _require_object(
            s3, bucket, output_prefix + "/" + required_prior_manifest,
            stage_code,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_digest = hashlib.sha1(
        inference_run.encode("utf-8")
    ).hexdigest()[:8]
    job_code = "ai-pal-{}-{}-{}".format(
        stage_code, case_code, run_digest
    )
    job_name = job_code + "-" + timestamp
    stage_prefix = (
        "sagemaker/ai-pal/inference/{}/jobs/{}/source"
        .format(inference_run, job_name)
    )
    with tempfile.TemporaryDirectory(prefix="ai-pal-stage-") as temp_dir:
        stage = Path(temp_dir) / "source"
        pal_stage = stage / "PAL_src"
        workflow_stage = stage / "workflow"
        input_stage = workflow_stage / "input"
        pal_stage.mkdir(parents=True)
        input_stage.mkdir(parents=True)
        shutil.copy2(requirements, stage / "requirements.txt")
        shutil.copy2(shared_config, pal_stage / "config_ai_pal.py")
        for name in PAL_SOURCE_NAMES:
            shutil.copy2(ai_pal_root / "PAL_src" / name, pal_stage / name)
        for path in station_paths:
            shutil.copy2(path, input_stage / path.name)
        if include_models:
            for model in MODEL_NAMES:
                lower = model.lower()
                shutil.copy2(
                    workflow_dir / "config_{}_{}.py".format(lower, case_code),
                    workflow_stage / "config_{}_case.py".format(lower),
                )
                if include_positive_models:
                    shutil.copy2(
                        workflow_dir / "config_{}_pos_ceed.py".format(lower),
                        workflow_stage / "config_{}_pos_ceed.py".format(lower),
                    )
                upload_tree(
                    s3, ai_pal_root / "picker_{}".format(model), bucket,
                    stage_prefix + "/picker_{}".format(model),
                )
            if include_positive_models:
                ceed_stage = input_stage / "CEED_ckpt"
                ceed_stage.mkdir()
                for name in POSITIVE_CHECKPOINTS:
                    shutil.copy2(
                        workflow_dir / "input" / "CEED_ckpt" / name,
                        ceed_stage / name,
                    )
        upload_tree(s3, stage, bucket, stage_prefix)

    inputs = [_input(
        "source", "s3://{}/{}/".format(bucket, stage_prefix),
        "/opt/ml/processing/source",
    )]
    if include_models:
        checkpoint_uri = checkpoint_s3_root_uri or (
            "s3://{}/sagemaker/ai-pal/training/{}/03_checkpoints/"
            .format(bucket, training_run)
        )
        checkpoint_bucket, checkpoint_prefix = _split_s3_uri(checkpoint_uri)
        for model in MODEL_NAMES:
            prefix = checkpoint_prefix + "/" + model
            if not prefix_has_objects(s3, checkpoint_bucket, prefix):
                raise FileNotFoundError(
                    "no {} checkpoints under s3://{}/{}".format(
                        model, checkpoint_bucket, prefix
                    )
                )
            inputs.append(_input(
                "checkpoint-" + model.lower(),
                "s3://{}/{}/".format(checkpoint_bucket, prefix),
                "/opt/ml/processing/checkpoints/" + model,
            ))
    if prefix_has_objects(s3, bucket, output_prefix):
        inputs.append(_input(
            "resume", output_uri, "/opt/ml/processing/resume"
        ))
        print("resuming from: {}".format(output_uri))

    env = {
        "CASE_CODE": case_code,
        "TIME_RANGE": time_range,
        "FULL_STATION_FILE": station_paths[0].name,
        "SUBNET_STATION_FILES": json.dumps({
            "r{}".format(index): path.name
            for index, path in enumerate(station_paths[1:], start=1)
        }),
        "MODEL_GPU_MAP": json.dumps(model_gpu_map),
        "NUM_WORKERS": str(num_workers),
        "OVERWRITE": "1" if overwrite else "0",
        "OUTPUT_S3_URI": output_uri,
        "SYNC_INTERVAL_SEC": str(sync_interval_sec),
        "SCEDC_ACCESS_MODE": "signed",
        "SCEDC_REGION": region,
        "OMP_NUM_THREADS": str(cpu_threads),
        "OPENBLAS_NUM_THREADS": str(cpu_threads),
        "MKL_NUM_THREADS": str(cpu_threads),
        "PYTHONPATH": "/opt/ml/processing/source:/opt/ml/processing/source/PAL_src",
    }
    env.update(extra_env or {})
    image_uri = retrieve(
        framework="pytorch", region=region, version=framework_version,
        py_version=python_version, instance_type=instance_type,
        image_scope="training",
    )
    processor = ScriptProcessor(
        image_uri=image_uri, command=["python3"], role=role,
        instance_type=instance_type, instance_count=instance_count,
        volume_size_in_gb=volume_size_gb,
        max_runtime_in_seconds=max_runtime_seconds,
        base_job_name=job_code, sagemaker_session=session, env=env,
    )
    processor.run(
        code=str(entry), inputs=inputs, outputs=[], job_name=job_name,
        wait=False, logs=False,
    )
    print("submitted: {}".format(job_name))
    print("stage:     {}".format(stage_code))
    print("output:    {}".format(output_uri))
    print("monitor:   python processing_job/monitor_staged_ai_pal_jobs.py")
