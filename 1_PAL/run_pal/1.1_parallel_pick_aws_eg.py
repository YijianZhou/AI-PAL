#!/usr/bin/env python3
"""Example parallel PAL picking launcher for the SCEDC AWS archive."""

import contextlib
import multiprocessing
import os
import sys
from datetime import timedelta
from pathlib import Path

# i/o paths
run_dir = Path(__file__).resolve().parent
pal_dir = Path(os.environ.get("PAL_DIR", "/home/zhouyj/software/1_PAL"))
sys.path.insert(0, str(pal_dir))
from run_pick_aws import parse_time_range, run_pick
import config_aws_eg
sta_file = (run_dir / "input" /
            "example_aws_pal.csv").resolve()
out_pick_dir = (run_dir / "output" / "aws_eg" / "picks").resolve()
log_dir = (run_dir / "output" / "aws_eg" / "logs").resolve()

# parallel and date params; end date is exclusive
time_range = "20250101-20250102"
num_workers = 1
overwrite = False

# S3 params
bucket = "scedc-pds"
region = "us-west-2"
root_prefix = "continuous_waveforms"
access_mode = "signed"
location_priority = ()
acceleration_instrument_codes = ("N",)

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


def pick_worker(worker_range, log_path):
    with open(log_path, "w") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_pick(
                worker_range, sta_file, out_pick_dir, pal_dir, cfg,
                bucket, region, root_prefix, access_mode,
                location_priority, acceleration_instrument_codes, overwrite,
            )


def main():
    start_date, end_date = parse_time_range(time_range)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for worker_idx, (start, end) in enumerate(
        split_ranges(start_date, end_date, num_workers), start=1
    ):
        worker_range = "{}-{}".format(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        log_path = log_dir / "pick_worker_{}_{}.log".format(worker_idx, worker_range)
        print("launch worker {}: {} -> {}".format(worker_idx, worker_range, log_path))
        process = multiprocessing.Process(
            target=pick_worker, args=(worker_range, log_path),
        )
        process.start()
        processes.append((worker_idx, worker_range, process, log_path))

    failures = []
    for worker_idx, worker_range, process, log_path in processes:
        process.join()
        print("worker {} finished with code {}: {}".format(
            worker_idx, process.exitcode, worker_range,
        ))
        if process.exitcode != 0:
            failures.append((worker_idx, process.exitcode, log_path))
    if failures:
        raise RuntimeError("pick workers failed: {}".format(failures))


if __name__ == "__main__":
    main()
