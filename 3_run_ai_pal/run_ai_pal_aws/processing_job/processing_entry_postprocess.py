#!/usr/bin/env python3
"""Container entry point for staged event repicking and reassociation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SOURCE_ROOT = Path("/opt/ml/processing/source")
PAL_SRC = SOURCE_ROOT / "PAL_src"
WORKFLOW = SOURCE_ROOT / "workflow"
CHECKPOINT_ROOT = Path("/opt/ml/processing/checkpoints")
RESUME_ROOT = Path("/opt/ml/processing/resume")
OUTPUT_ROOT = Path("/opt/ml/processing/output")


def main():
    requirements = SOURCE_ROOT / "requirements.txt"
    if requirements.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        )
    for path in (SOURCE_ROOT, PAL_SRC):
        sys.path.insert(0, str(path))

    from aws_inference_job import checkpoint_file, prepare_output
    import config_ai_pal as config_module
    from offline_event_postprocessor import run_offline_event_postprocessing

    case_code = os.environ["CASE_CODE"]
    time_range = os.environ["TIME_RANGE"]
    station_file = WORKFLOW / "input" / os.environ["FULL_STATION_FILE"]
    writer = prepare_output(
        OUTPUT_ROOT, RESUME_ROOT, os.environ["OUTPUT_S3_URI"],
        stale_files=(
            "2.3_postprocess_manifest.json", "2.3_postprocess_failure.json",
        ),
    )
    cfg = config_module.Config()
    cfg.enable_event_waveform_plot = (
        os.environ.get("ENABLE_EVENT_WAVEFORM_PLOT", "0") == "1"
    )
    cfg.save_filtered_event_waveforms = (
        os.environ.get("SAVE_FILTERED_EVENT_WAVEFORMS", "0") == "1"
    )

    def event_waveform_complete(_origin, event_dir):
        for path in Path(event_dir).rglob("*"):
            writer.upload_file(path)

    def day_complete(_date, _phase_path, _status_path, _summary):
        writer.sync()

    cfg.event_waveform_complete_callback = event_waveform_complete
    gpu_map = json.loads(os.environ.get("MODEL_GPU_MAP", "{}"))
    positive_names = {
        "SAR": "ceed_pos_sar_best.ckpt", "FT": "ceed_pos_ft_best.ckpt",
        "PHN": "ceed_pos_phn_best.ckpt", "RUN": "ceed_pos_run_best.ckpt",
    }
    pos_neg = {
        model: {
            "config": WORKFLOW / "config_{}_case.py".format(model.lower()),
            "gpu_idx": int(gpu_map.get(model, 0)),
            "ckpt": checkpoint_file(CHECKPOINT_ROOT / model / "best.ckpt"),
        }
        for model in dict.fromkeys(cfg.repicker_pos_neg_group)
    }
    positive = {
        model: {
            "config": WORKFLOW / "config_{}_pos_ceed.py".format(model.lower()),
            "gpu_idx": int(gpu_map.get(model, 0)),
            "ckpt": WORKFLOW / "input" / "CEED_ckpt" / positive_names[model],
        }
        for model in dict.fromkeys(cfg.repicker_pos_group)
    }
    case_root = OUTPUT_ROOT / case_code
    initial_phase_dir = (
        case_root / "2.1.0_phase_init_AI-PAL" / "daily_assoc" / "merged"
    )
    final_root = case_root / "3.1_phase_final_AI-PAL"
    writer.exclude_from_sync(final_root / "event_waveforms")
    started = time.time()
    writer.write_json("_status/2.3_postprocess_job.json", {
        "status": "running", "time_range": time_range,
        "save_filtered_event_waveforms": cfg.save_filtered_event_waveforms,
        "enable_event_waveform_plot": cfg.enable_event_waveform_plot,
    })
    try:
        paths = run_offline_event_postprocessing(
            ai_pal_root=SOURCE_ROOT,
            cfg=cfg,
            repicker_pos_neg_specs=pos_neg,
            repicker_pos_specs=positive,
            data_dir=station_file,
            station_file=station_file,
            initial_phase_dir=initial_phase_dir,
            final_root=final_root,
            time_range=time_range,
            num_workers=int(os.environ.get("NUM_WORKERS", "12")),
            overwrite=os.environ.get("OVERWRITE", "0") == "1",
            day_complete_callback=day_complete,
        )
        writer.sync()
        writer.write_json("2.3_postprocess_manifest.json", {
            "status": "complete", "time_range": time_range,
            "num_daily_phase_files": len(paths),
            "save_filtered_event_waveforms": cfg.save_filtered_event_waveforms,
            "elapsed_sec": time.time() - started,
        })
    except Exception as exc:
        writer.write_json("2.3_postprocess_failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        })
        writer.sync()
        raise


if __name__ == "__main__":
    main()
