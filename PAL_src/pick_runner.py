"""Parallel runner for local and SCEDC AWS daily PAL picking."""

import contextlib
import importlib
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path


def parse_date_range(time_range):
    start_text, end_text = time_range.split("-")
    start = datetime.strptime(start_text, "%Y%m%d").date()
    end = datetime.strptime(end_text, "%Y%m%d").date()
    if start >= end:
        raise ValueError("time_range must have start < exclusive end")
    return start, end


def _split_ranges(start, end, worker_count):
    num_days = (end - start).days
    worker_count = min(max(1, worker_count), num_days)
    base_days, remainder = divmod(num_days, worker_count)
    ranges = []
    current = start
    for worker_index in range(worker_count):
        days = base_days + (1 if worker_index < remainder else 0)
        next_date = current + timedelta(days=days)
        ranges.append((current, next_date))
        current = next_date
    return ranges


def _pick_worker(task):
    from run_pick import run_pick

    (
        worker_range,
        data_dir,
        station_file,
        pick_dir,
        config_module,
        config_class,
        overwrite,
        log_path,
    ) = task
    cfg = getattr(importlib.import_module(config_module), config_class)()
    with Path(log_path).open("w", encoding="utf-8") as log_fp:
        with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
            run_pick(
                worker_range,
                data_dir,
                station_file,
                pick_dir,
                cfg,
                overwrite=overwrite,
            )


def run_parallel_local_pick(
    time_range,
    data_dir,
    station_file,
    pick_dir,
    log_dir,
    num_workers,
    config_factory,
    overwrite=False,
    include_association_halo=False,
):
    """Write daily picks, optionally including one day beyond each boundary."""
    start, end = parse_date_range(time_range)
    if include_association_halo:
        start -= timedelta(days=1)
        end += timedelta(days=1)
    if start >= end:
        raise ValueError("time range must contain at least one day")

    pick_dir = Path(pick_dir).resolve()
    log_dir = Path(log_dir).resolve()
    pick_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, (worker_start, worker_end) in enumerate(
        _split_ranges(start, end, num_workers), start=1
    ):
        worker_range = "{}-{}".format(
            worker_start.strftime("%Y%m%d"),
            worker_end.strftime("%Y%m%d"),
        )
        log_path = log_dir / "pick_worker_{}_{}.log".format(index, worker_range)
        print("launch pick worker {}: {} -> {}".format(index, worker_range, log_path))
        tasks.append((
            worker_range,
            str(Path(data_dir).resolve()),
            str(Path(station_file).resolve()),
            str(pick_dir),
            config_factory.__module__,
            config_factory.__name__,
            bool(overwrite),
            str(log_path),
        ))

    with multiprocessing.Pool(processes=len(tasks)) as pool:
        results = [pool.apply_async(_pick_worker, (task,)) for task in tasks]
        failures = []
        for index, result in enumerate(results, start=1):
            try:
                result.get()
                print("pick worker {} completed".format(index))
            except Exception as exc:
                failures.append((index, repr(exc)))
    if failures:
        raise RuntimeError("pick workers failed: {}".format(failures))

def _aws_pick_worker(task):
    (
        worker_range,
        station_file,
        pick_dir,
        pal_source_dir,
        config_module,
        config_class,
        bucket,
        region,
        root_prefix,
        access_mode,
        location_priority,
        acceleration_instrument_codes,
        overwrite,
        retry_failed_dates,
        log_path,
    ) = task
    from run_pick_aws import run_pick as run_pick_aws

    cfg = getattr(importlib.import_module(config_module), config_class)()
    with Path(log_path).open("w", encoding="utf-8") as log_fp:
        with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
            run_pick_aws(
                worker_range,
                station_file,
                pick_dir,
                pal_source_dir,
                cfg,
                bucket,
                region,
                root_prefix,
                access_mode,
                location_priority,
                acceleration_instrument_codes,
                overwrite,
                retry_failed_dates,
            )


def run_parallel_aws_pick(
    time_range,
    station_file,
    pick_dir,
    log_dir,
    pal_source_dir,
    num_workers,
    config_factory,
    bucket="scedc-pds",
    region="us-west-2",
    root_prefix="continuous_waveforms",
    access_mode="signed",
    location_priority=(),
    acceleration_instrument_codes=("N",),
    overwrite=False,
    retry_failed_dates=False,
):
    """Run independent daily AWS picking ranges in parallel processes."""
    start, end = parse_date_range(time_range)
    pick_dir = Path(pick_dir).resolve()
    log_dir = Path(log_dir).resolve()
    pick_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for index, (worker_start, worker_end) in enumerate(
        _split_ranges(start, end, num_workers), start=1
    ):
        worker_range = "{}-{}".format(
            worker_start.strftime("%Y%m%d"),
            worker_end.strftime("%Y%m%d"),
        )
        log_path = log_dir / "pick_worker_{}_{}.log".format(index, worker_range)
        print("launch pick worker {}: {} -> {}".format(
            index, worker_range, log_path
        ))
        tasks.append((
            worker_range,
            str(Path(station_file).resolve()),
            str(pick_dir),
            str(Path(pal_source_dir).expanduser().resolve()),
            config_factory.__module__,
            config_factory.__name__,
            bucket,
            region,
            root_prefix,
            access_mode,
            tuple(location_priority),
            tuple(acceleration_instrument_codes),
            bool(overwrite),
            bool(retry_failed_dates),
            str(log_path),
        ))

    with multiprocessing.Pool(processes=len(tasks)) as pool:
        results = [pool.apply_async(_aws_pick_worker, (task,)) for task in tasks]
        failures = []
        for index, result in enumerate(results, start=1):
            try:
                result.get()
                print("pick worker {} completed".format(index))
            except Exception as exc:
                failures.append((index, repr(exc)))
    if failures:
        raise RuntimeError("AWS pick workers failed: {}".format(failures))






