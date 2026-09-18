#!/usr/bin/env python3
"""Submit stage 2.2: buffered PAL association and subnet merging."""

from pathlib import Path

from processing_job.submit_stage import submit_stage


# ============================================================================
# USER SETTINGS
# ============================================================================
CASE_CODE = "eg"
TIME_RANGE = "20190704-20190707"
INFERENCE_RUN = "eg-ai-pal-staged-20190704-20190707-v1"
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()
WORKFLOW_DIR = Path(__file__).resolve().parent
FULL_STATION_FILE = Path("input/example_pal_format1.sta")
SUBNET_STATION_FILES = []  # Empty means one full-network association.

instance_type = "ml.c5.4xlarge"
instance_count = 1
volume_size_gb = 256
max_runtime_seconds = 432000
num_workers = 16
cpu_threads = 1
overwrite = False


if __name__ == "__main__":
    submit_stage(
        workflow_dir=WORKFLOW_DIR, ai_pal_root=AI_PAL_ROOT,
        case_code=CASE_CODE, time_range=TIME_RANGE,
        inference_run=INFERENCE_RUN, training_run="",
        checkpoint_s3_root_uri=None,
        full_station_file=FULL_STATION_FILE,
        subnet_station_files=SUBNET_STATION_FILES,
        stage_code="2-2-assoc", entry_name="processing_entry_assoc.py",
        required_prior_manifest="2.1_pick_manifest.json",
        include_models=False,
        instance_type=instance_type, instance_count=instance_count,
        volume_size_gb=volume_size_gb,
        max_runtime_seconds=max_runtime_seconds, num_workers=num_workers,
        model_gpu_map={}, overwrite=overwrite, cpu_threads=cpu_threads,
    )
