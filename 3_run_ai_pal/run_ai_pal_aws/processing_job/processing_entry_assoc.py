#!/usr/bin/env python3
"""Container entry point for staged buffered PAL association."""

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

    from association_runner import run_buffered_association
    from aws_inference_job import PeriodicS3Sync, prepare_output
    import config_ai_pal as config_module

    case_code = os.environ["CASE_CODE"]
    time_range = os.environ["TIME_RANGE"]
    full_station = WORKFLOW / "input" / os.environ["FULL_STATION_FILE"]
    subnet_names = json.loads(os.environ.get("SUBNET_STATION_FILES", "{}"))
    station_sets = {
        name: WORKFLOW / "input" / filename
        for name, filename in subnet_names.items()
    }
    if not station_sets:
        station_sets = {"full": full_station}
    for path in station_sets.values():
        if not path.exists():
            raise FileNotFoundError(path)
    writer = prepare_output(
        OUTPUT_ROOT, RESUME_ROOT, os.environ["OUTPUT_S3_URI"],
        stale_files=("2.2_assoc_manifest.json", "2.2_assoc_failure.json"),
    )
    cfg = config_module.Config()
    case_root = OUTPUT_ROOT / case_code
    pick_dir = case_root / "1.2_picks_AI-PAL-ENSEMBLE"
    initial_root = case_root / "2.1.0_phase_init_AI-PAL"
    started = time.time()
    writer.write_json("_status/2.2_assoc_job.json", {
        "status": "running", "time_range": time_range,
        "station_sets": sorted(station_sets),
    })
    try:
        with PeriodicS3Sync(writer, os.environ.get("SYNC_INTERVAL_SEC", "300")):
            run_buffered_association(
                subnet_station_files=station_sets,
                pick_dir=pick_dir,
                assoc_root=initial_root / "daily_assoc",
                time_range=time_range,
                num_workers=int(os.environ.get("NUM_WORKERS", "16")),
                config_factory=config_module.Config,
                overwrite=os.environ.get("OVERWRITE", "0") == "1",
                output_catalog=initial_root / "catalog_{}.dat".format(time_range),
                output_phase=initial_root / "phase_{}.dat".format(time_range),
            )
        writer.write_json("2.2_assoc_manifest.json", {
            "status": "complete", "time_range": time_range,
            "elapsed_sec": time.time() - started,
        })
    except Exception as exc:
        writer.write_json("2.2_assoc_failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        })
        writer.sync()
        raise


if __name__ == "__main__":
    main()
