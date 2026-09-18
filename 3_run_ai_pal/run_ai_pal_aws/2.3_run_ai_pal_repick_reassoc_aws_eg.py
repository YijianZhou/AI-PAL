#!/usr/bin/env python3
"""Submit stage 2.3: event repicking, reassociation, and waveform export."""

from pathlib import Path

from processing_job.submit_stage import submit_stage


# ============================================================================
# USER SETTINGS
# ============================================================================
CASE_CODE = "eg"
TIME_RANGE = "20190704-20190707"
INFERENCE_RUN = "eg-ai-pal-staged-20190704-20190707-v1"
TRAINING_RUN = "eg-2020-2025-rarity-v1"
CHECKPOINT_S3_ROOT_URI = None
AI_PAL_ROOT = Path("~/shared/software/AI-PAL").expanduser()
WORKFLOW_DIR = Path(__file__).resolve().parent
FULL_STATION_FILE = Path("input/example_pal_format1.sta")

# SAC files are written under 3.1_phase_final_AI-PAL/event_waveforms and can
# be consumed by the cross-correlation relocation preprocessing workflow.
SAVE_FILTERED_EVENT_WAVEFORMS = False
ENABLE_EVENT_WAVEFORM_PLOT = False

MODEL_GPU_MAP = {"SAR": 0, "FT": 1, "PHN": 2, "RUN": 3}
instance_type = "ml.g4dn.12xlarge"
instance_count = 1
volume_size_gb = 1024
max_runtime_seconds = 432000
num_workers = 12
cpu_threads = 1
overwrite = False


if __name__ == "__main__":
    submit_stage(
        workflow_dir=WORKFLOW_DIR, ai_pal_root=AI_PAL_ROOT,
        case_code=CASE_CODE, time_range=TIME_RANGE,
        inference_run=INFERENCE_RUN, training_run=TRAINING_RUN,
        checkpoint_s3_root_uri=CHECKPOINT_S3_ROOT_URI,
        full_station_file=FULL_STATION_FILE, subnet_station_files=[],
        stage_code="2-3-postprocess",
        entry_name="processing_entry_postprocess.py",
        required_prior_manifest="2.2_assoc_manifest.json",
        include_models=True,
        instance_type=instance_type, instance_count=instance_count,
        volume_size_gb=volume_size_gb,
        max_runtime_seconds=max_runtime_seconds, num_workers=num_workers,
        model_gpu_map=MODEL_GPU_MAP, overwrite=overwrite,
        cpu_threads=cpu_threads,
        extra_env={
            "SAVE_FILTERED_EVENT_WAVEFORMS": (
                "1" if SAVE_FILTERED_EVENT_WAVEFORMS else "0"
            ),
            "ENABLE_EVENT_WAVEFORM_PLOT": (
                "1" if ENABLE_EVENT_WAVEFORM_PLOT else "0"
            ),
        },
    )
