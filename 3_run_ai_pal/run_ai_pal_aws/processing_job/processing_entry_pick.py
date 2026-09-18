#!/usr/bin/env python3
"""Container entry point for staged AI-PAL daily picking."""

import json
import os
from datetime import timedelta
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

    from association_runner import parse_date_range
    from aws_inference_job import (
        PeriodicS3Sync, checkpoint_file, prepare_output,
    )
    import config_ai_pal as config_module
    from offline_picker_runner import run_offline_picker_ensemble

    case_code = os.environ["CASE_CODE"]
    target_range = os.environ["TIME_RANGE"]
    station_file = WORKFLOW / "input" / os.environ["FULL_STATION_FILE"]
    if not station_file.exists():
        raise FileNotFoundError(station_file)
    writer = prepare_output(
        OUTPUT_ROOT, RESUME_ROOT, os.environ["OUTPUT_S3_URI"],
        stale_files=("2.1_pick_manifest.json", "2.1_pick_failure.json"),
    )
    cfg = config_module.Config()
    gpu_map = json.loads(os.environ.get("MODEL_GPU_MAP", "{}"))
    specs = {}
    for model in dict.fromkeys(cfg.picker_pos_neg_group):
        specs[model] = {
            "group": "POS_NEG", "model": model,
            "config": WORKFLOW / "config_{}_case.py".format(model.lower()),
            "gpu_idx": int(gpu_map.get(model, 0)),
            "ckpt": checkpoint_file(CHECKPOINT_ROOT / model / "best.ckpt"),
        }
    start, end = parse_date_range(target_range)
    pick_range = "{}-{}".format(
        (start - timedelta(days=1)).strftime("%Y%m%d"),
        (end + timedelta(days=1)).strftime("%Y%m%d"),
    )
    case_root = OUTPUT_ROOT / case_code
    ensemble_dir = case_root / "1.2_picks_AI-PAL-ENSEMBLE"
    started = time.time()
    writer.write_json("_status/2.1_pick_job.json", {
        "status": "running", "target_time_range": target_range,
        "pick_time_range": pick_range,
    })
    try:
        with PeriodicS3Sync(writer, os.environ.get("SYNC_INTERVAL_SEC", "300")):
            summaries = run_offline_picker_ensemble(
                ai_pal_root=SOURCE_ROOT,
                cfg=cfg,
                picker_specs=specs,
                data_dir=station_file,
                station_file=station_file,
                time_range=pick_range,
                individual_pick_root=case_root,
                ensemble_pick_dir=ensemble_dir,
                num_workers=int(os.environ.get("NUM_WORKERS", "8")),
                overwrite=os.environ.get("OVERWRITE", "0") == "1",
            )
        writer.write_json("2.1_pick_manifest.json", {
            "status": "complete", "target_time_range": target_range,
            "pick_time_range": pick_range, "num_pick_days": len(summaries),
            "elapsed_sec": time.time() - started,
        })
    except Exception as exc:
        writer.write_json("2.1_pick_failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        })
        writer.sync()
        raise


if __name__ == "__main__":
    main()
