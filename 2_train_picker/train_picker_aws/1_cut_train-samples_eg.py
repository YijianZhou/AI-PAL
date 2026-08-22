#!/usr/bin/env python3
"""Stage inputs and submit one yearly SCEDC NPY sample-cutting job."""

import csv
import shutil
import tempfile
from datetime import date, datetime, timedelta, timezone
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

from processing_job.job_common import (
    delete_s3_prefix, prefix_has_objects, upload_tree,
)

# ============================================================================
# USER SETTINGS: edit paths, case identity, AWS resources, and runtime here
# ============================================================================

# Case configs use config_ai_pal_<case>.py and config_<model>_<case>.py.
CASE_CODE = "eg"  # Example case; change consistently for another workflow.

# Local source and prepared input paths
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()  # Installed source package.
workflow_dir = Path(__file__).resolve().parent  # Copied case workflow.
preprocess_dir = AI_PAL_ROOT / "picker_SAR" / "preprocess"
pal_data_pipeline = AI_PAL_ROOT / "PAL_src" / "data_pipeline.py"
pal_data_pipeline_aws = AI_PAL_ROOT / "PAL_src" / "data_pipeline_aws.py"
local_config = workflow_dir / "config_ai_pal_{}.py".format(CASE_CODE)
training_adapter = AI_PAL_ROOT / "PAL_src" / "data_pipeline_training_aws.py"
station_file = "station_scedc_aws_selected_20200101_20260701_pal.csv"

# Submit one year at a time. Change only this field for the next yearly job.
training_year = 2020
year_input_dir = workflow_dir / "input" / str(training_year)
phase_file = year_input_dir / (
    "%s_assoc_%d_pal_rarity.pha" % (CASE_CODE, training_year)
)
association_rate_file = year_input_dir / (
    "%s_assoc_%d_association_rates.csv" % (CASE_CODE, training_year)
)
station_path = workflow_dir / "input" / station_file

# Reusable annual NPY sample-library identity and output location. Keep this
# unchanged for every year that may be combined into a later Zarr dataset.
region = "us-west-2"
SAMPLE_RUN = "%s-2020-2025-v1" % CASE_CODE
job_code = "ai-pal-cut-{}-{}".format(SAMPLE_RUN, training_year)
artifact_prefix = "sagemaker/ai-pal/training/" + SAMPLE_RUN
overwrite_existing_output = False

# SCEDC waveform access
scedc_bucket = "scedc-pds"
scedc_region = "us-west-2"
scedc_root_prefix = "continuous_waveforms"
scedc_access_mode = "signed"
acceleration_codes = "N"

# CPU Processing controls
instance_type = "ml.c5.4xlarge"
instance_count = 1
volume_size_gb = 200
max_runtime_seconds = 432000
num_workers = 8
shard_size = 1024
threads_per_worker = 3

# Managed PyTorch CPU image
framework_version = "2.6.0"
python_version = "py312"


# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

def dates_in_year(year):
    current = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    while current < end:
        yield current
        current += timedelta(days=1)


def processing_input(name, uri, local_path):
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=uri,
            local_path=local_path,
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    )


def validate_year_inputs():
    required = (
        local_config,
        training_adapter,
        pal_data_pipeline,
        pal_data_pipeline_aws,
        preprocess_dir,
        station_path,
        phase_file,
        association_rate_file,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    expected_dates = {value.isoformat() for value in dates_in_year(training_year)}
    required_columns = {
        "date", "net_sta", "num_picks",
        "num_associated_picks", "num_unassociated_picks",
    }
    observed_dates = set()
    with association_rate_file.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "{} missing columns: {}".format(
                    association_rate_file, ", ".join(sorted(missing_columns))
                )
            )
        for row in reader:
            if row["date"].strip().lower() == "date":
                continue
            observed_dates.add(row["date"].strip())
    missing_dates = sorted(expected_dates - observed_dates)
    unexpected_dates = sorted(observed_dates - expected_dates)
    if not observed_dates:
        raise ValueError("{} contains no station-date rows".format(
            association_rate_file
        ))
    if unexpected_dates:
        raise ValueError(
            "{} contains {} dates outside {}".format(
                association_rate_file, len(unexpected_dates), training_year
            )
        )


def main():
    validate_year_inputs()
    processing_dir = Path(__file__).resolve().parent / "processing_job"
    entry = processing_dir / "processing_entry_cut_samples.py"
    requirements = processing_dir / "requirements.txt"
    for path in (entry, requirements):
        if not path.exists():
            raise FileNotFoundError(path)

    session = Session(boto_session=boto3.Session(region_name=region))
    role = get_execution_role()
    bucket = session.default_bucket()
    s3 = boto3.client("s3", region_name=region)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = job_code + "-" + timestamp
    stage_root = artifact_prefix + "/jobs/" + job_name
    source_prefix = stage_root + "/source"
    input_prefix = stage_root + "/input"
    output_prefix = artifact_prefix + "/01_npy/{}".format(training_year)
    output_uri = "s3://{}/{}/".format(bucket, output_prefix)

    output_exists = prefix_has_objects(s3, bucket, output_prefix)
    if output_exists and not overwrite_existing_output:
        raise FileExistsError(
            "annual output already exists at {}; remove it or set "
            "overwrite_existing_output=True to delete and replace it".format(
                output_uri
            )
        )
    if output_exists:
        deleted = delete_s3_prefix(s3, bucket, output_prefix)
        print(
            "deleted {} existing output object(s) from {}".format(
                deleted, output_uri
            ),
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="ai-pal-cut-") as temp_dir:
        stage = Path(temp_dir)
        source_stage = stage / "source"
        input_stage = stage / "input"
        shutil.copytree(
            preprocess_dir,
            source_stage,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(local_config, source_stage / "config.py")
        shutil.copy2(training_adapter, source_stage)
        shutil.copy2(pal_data_pipeline, source_stage)
        shutil.copy2(pal_data_pipeline_aws, source_stage)
        shutil.copy2(requirements, source_stage / "requirements.txt")

        (input_stage / "phase").mkdir(parents=True)
        (input_stage / "association_rate").mkdir(parents=True)
        (input_stage / "station").mkdir(parents=True)
        shutil.copy2(phase_file, input_stage / "phase" / phase_file.name)
        shutil.copy2(
            association_rate_file,
            input_stage / "association_rate" / association_rate_file.name,
        )
        shutil.copy2(station_path, input_stage / "station" / station_file)

        upload_tree(s3, source_stage, bucket, source_prefix)
        upload_tree(s3, input_stage, bucket, input_prefix)

    source_uri = "s3://{}/{}/".format(bucket, source_prefix)
    input_uri = "s3://{}/{}/".format(bucket, input_prefix)
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
            "TRAINING_YEAR": str(training_year),
            "NUM_WORKERS": str(num_workers),
            "SHARD_SIZE": str(shard_size),
            "SCEDC_BUCKET": scedc_bucket,
            "SCEDC_REGION": scedc_region,
            "SCEDC_ROOT_PREFIX": scedc_root_prefix,
            "SCEDC_ACCESS_MODE": scedc_access_mode,
            "ACCELERATION_CODES": acceleration_codes,
            "OMP_NUM_THREADS": str(threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(threads_per_worker),
            "MKL_NUM_THREADS": str(threads_per_worker),
            "NUMEXPR_NUM_THREADS": str(threads_per_worker),
            "OMP_DYNAMIC": "FALSE",
        },
    )
    processor.run(
        code=str(entry),
        inputs=[
            processing_input(
                "source", source_uri, "/opt/ml/processing/source"
            ),
            processing_input(
                "year-input", input_uri, "/opt/ml/processing/input"
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="npy-samples",
                s3_output=ProcessingS3Output(
                    s3_uri=output_uri,
                    local_path="/opt/ml/processing/output/npy",
                    s3_upload_mode="Continuous",
                ),
            )
        ],
        job_name=job_name,
        wait=False,
        logs=False,
    )
    print("submitted: " + job_name)
    print("year:      {}".format(training_year))
    print("waveforms: s3://{}/{}/".format(scedc_bucket, scedc_root_prefix))
    print("output:    " + output_uri)
    print("monitor:   python processing_job/monitor_cut_samples_job.py")


if __name__ == "__main__":
    main()
