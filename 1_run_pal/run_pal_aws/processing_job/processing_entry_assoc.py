#!/usr/bin/env python3
"""Processing-container entry point for one buffered PAL association job."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


WORK_DIR = Path("/opt/ml/processing/work")
PAL_DIR = Path("/opt/ml/processing/pal")
PICK_INPUT_DIR = Path("/opt/ml/processing/picks")
RESUME_DIR = Path("/opt/ml/processing/resume")
RUNTIME_FILE = WORK_DIR / "assoc_runtime.json"
REQUIREMENTS_FILE = WORK_DIR / "requirements.txt"


def main():
    if REQUIREMENTS_FILE.exists():
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(REQUIREMENTS_FILE),
        ])

    runtime = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
    case_code = runtime["case_code"]
    output_dir = WORK_DIR / runtime["out_root"]
    if RESUME_DIR.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(RESUME_DIR, output_dir, dirs_exist_ok=True)

    pick_dir = WORK_DIR / runtime["pick_dir"]
    if pick_dir.exists() or pick_dir.is_symlink():
        if pick_dir.is_symlink() or pick_dir.is_file():
            pick_dir.unlink()
        else:
            shutil.rmtree(pick_dir)
    pick_dir.mkdir(parents=True)

    num_pick_files = 0
    for source in sorted(PICK_INPUT_DIR.rglob("*.pick")):
        target = pick_dir / source.name
        if target.exists():
            raise RuntimeError(
                "duplicate pick date from Processing inputs: {}".format(
                    source.name
                )
            )
        target.symlink_to(source.resolve())
        num_pick_files += 1
    if num_pick_files == 0:
        raise FileNotFoundError(
            "no .pick files under {}".format(PICK_INPUT_DIR)
        )
    print("linked {} daily pick files".format(num_pick_files), flush=True)

    sys.path.insert(0, str(PAL_DIR))
    sys.path.insert(0, str(WORK_DIR))
    config_module = importlib.import_module(
        "config_aws_%s" % case_code
    )
    from association_runner import run_buffered_association

    def work_path(value):
        path = Path(value)
        return path if path.is_absolute() else WORK_DIR / path

    run_buffered_association(
        subnet_station_files={
            name: work_path(value)
            for name, value in runtime["subnet_station_files"].items()
        },
        pick_dir=pick_dir,
        assoc_root=output_dir,
        time_range=runtime["time_range"],
        num_workers=int(runtime["num_workers"]),
        config_factory=config_module.Config,
        overwrite=bool(runtime["overwrite"]),
        retry_failed_days=bool(runtime["retry_failed_days"]),
    )


if __name__ == "__main__":
    main()
