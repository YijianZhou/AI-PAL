#!/usr/bin/env python3
"""Example parallel PAL association launcher for one SCEDC AWS subnet."""

import contextlib
import multiprocessing
import os
import sys
from datetime import timedelta
from pathlib import Path

# subnet and i/o paths: revise subnet_idx for each separate association run
run_dir = Path(__file__).resolve().parent
subnet_idx = 1
pal_dir = Path(os.environ.get("PAL_DIR", "/home/zhouyj/software/1_PAL"))
sys.path.insert(0, str(pal_dir))
from run_assoc_aws import run_assoc
from run_pick_aws import parse_time_range
import config_aws_eg
sta_file = run_dir / "input" / (
    "example_aws_pal_r{}.csv".format(subnet_idx)
)
pick_dir = run_dir / "output" / "aws_eg" / "picks"
out_root = run_dir / "output" / "aws_eg{}".format(subnet_idx)
log_dir = run_dir / "logs" / "assoc" / "aws_eg{}".format(subnet_idx)

# parallel and date params; end date is exclusive
time_range = "20250101-20250102"
num_workers = 1

# PAL model and data-pipeline config
cfg = config_aws_eg.Config()


def split_ranges(start, end, worker_count):
    num_days = (end - start).days
    worker_count = min(max(1, worker_count), num_days)
    base_days, remainder = divmod(num_days, worker_count)
    ranges = []
    current = start
    for worker_idx in range(worker_count):
        days = base_days + (1 if worker_idx < remainder else 0)
        next_date = current + timedelta(days=days)
        ranges.append((current, next_date))
        current = next_date
    return ranges

def assoc_worker(worker_range, output_catalog, output_phase, log_path):
    with open(log_path, "w") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_assoc(
                worker_range, sta_file, pick_dir,
                output_catalog, output_phase, pal_dir, cfg,
            )


def main():
    if not sta_file.exists():
        raise FileNotFoundError(sta_file)
    start_date, end_date = parse_time_range(time_range)
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for worker_idx, (start, end) in enumerate(
        split_ranges(start_date, end_date, num_workers), start=1
    ):
        worker_range = "{}-{}".format(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        output_catalog = out_root / ("catalog_" + worker_range + ".dat")
        output_phase = out_root / ("phase_" + worker_range + ".dat")
        log_path = log_dir / "assoc_worker_{}_{}.log".format(worker_idx, worker_range)
        print("launch aws_eg{} worker {}: {} -> {}".format(
            subnet_idx, worker_idx, worker_range, log_path,
        ))
        process = multiprocessing.Process(
            target=assoc_worker,
            args=(worker_range, output_catalog, output_phase, log_path),
        )
        process.start()
        processes.append((worker_idx, worker_range, process, log_path))

    failures = []
    for worker_idx, worker_range, process, log_path in processes:
        process.join()
        print("aws_eg{} worker {} finished with code {}: {}".format(
            subnet_idx, worker_idx, process.exitcode, worker_range,
        ))
        if process.exitcode != 0:
            failures.append((worker_idx, process.exitcode, log_path))
    if failures:
        raise RuntimeError("aws_eg{} association workers failed: {}".format(
            subnet_idx, failures,
        ))


if __name__ == "__main__":
    main()
