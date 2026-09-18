#!/usr/bin/env python3
"""Submit stage 2.1: buffered daily AI-PAL picking."""

from pathlib import Path

from processing_job.submit_stage import submit_stage


# ============================================================================
# USER SETTINGS
# ============================================================================
CASE_CODE = "eg"
TIME_RANGE = "20190704-20190707"  # Target dates; exclusive end date.
INFERENCE_RUN = "eg-ai-pal-staged-20190704-20190707-v1"
TRAINING_RUN = "eg-2020-2025-rarity-v1"
CHECKPOINT_S3_ROOT_URI = None
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()
WORKFLOW_DIR = Path(__file__).resolve().parent
FULL_STATION_FILE = Path("input/example_pal_format1.sta")

MODEL_GPU_MAP = {"SAR": 0, "FT": 0, "PHN": 0, "RUN": 0}
instance_type = "ml.g4dn.2xlarge"
instance_count = 1
volume_size_gb = 256
max_runtime_seconds = 432000
num_workers = 8
cpu_threads = 1
overwrite = False


if __name__ == "__main__":
    submit_stage(
        workflow_dir=WORKFLOW_DIR, ai_pal_root=AI_PAL_ROOT,
        case_code=CASE_CODE, time_range=TIME_RANGE,
        inference_run=INFERENCE_RUN, training_run=TRAINING_RUN,
        checkpoint_s3_root_uri=CHECKPOINT_S3_ROOT_URI,
        full_station_file=FULL_STATION_FILE, subnet_station_files=[],
        stage_code="2-1-pick", entry_name="processing_entry_pick.py",
        required_prior_manifest=None, include_models=True, include_positive_models=False,
        instance_type=instance_type, instance_count=instance_count,
        volume_size_gb=volume_size_gb,
        max_runtime_seconds=max_runtime_seconds, num_workers=num_workers,
        model_gpu_map=MODEL_GPU_MAP, overwrite=overwrite,
        cpu_threads=cpu_threads,
    )
