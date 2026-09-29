"""Spawn independent date blocks for the pick-only AI ensemble workflow."""
import contextlib
from datetime import datetime
import importlib.util
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback

from local_pick_blocks import date_blocks


def load_workflow_config(path):
    """Bind a process-private config without rewriting installed source files."""
    spec = importlib.util.spec_from_file_location("config_ai_pal", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["config_ai_pal"] = module
    spec.loader.exec_module(module)
    return module.Config()


def _worker(index, kwargs, config_path, log_path, messages):
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                cfg = load_workflow_config(config_path)
                from offline_picker_runner import run_offline_picker_ensemble
                def complete(day, path, summary):
                    import torch
                    memory = {}
                    for device in sorted({int(s["gpu_idx"]) for s in kwargs["picker_specs"].values()
                                          if int(s["gpu_idx"]) >= 0}):
                        memory[device] = {
                            "peak_allocated_MiB": round(torch.cuda.max_memory_allocated(device) / 2**20),
                            "peak_reserved_MiB": round(torch.cuda.max_memory_reserved(device) / 2**20),
                        }
                    skipped = bool(summary.get("skipped_existing", False))
                    print("GPU memory peaks (PyTorch only): {}".format(memory), flush=True)
                    messages.put(("day", index, (str(day.date), skipped, memory)))
                run_offline_picker_ensemble(cfg=cfg, day_complete_callback=complete, **kwargs)
                messages.put(("done", index, None))
            except BaseException as exc:
                traceback.print_exc(file=log)
                log.flush()
                messages.put(("error", index, str(exc)))
                raise


def run_offline_pick_blocks(*, ai_pal_root, config_path, picker_specs, data_dir,
                            station_file, time_range, individual_pick_root,
                            ensemble_pick_dir, log_dir, num_workers=2,
                            preprocessing_workers=4,
                            overwrite=False):
    if int(num_workers) < 1 or int(preprocessing_workers) < 1:
        raise ValueError("worker counts must be positive")
    start, end = [datetime.strptime(value, "%Y%m%d").date()
                  for value in time_range.split("-")]
    if start >= end:
        raise ValueError("time_range must have start < exclusive end")
    blocks = date_blocks(start, end, num_workers)
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    # Legacy branch-directory migrations must happen once, before workers start.
    for index, (name, spec) in enumerate(picker_specs.items(), 1):
        model = spec.get("identity", spec.get("model", name))
        root = Path(individual_pick_root)
        old = root / ("picks_" + name)
        new = root / "1.1.{}_picks_{}_{}".format(index, spec.get("group", "Local").lower(), model)
        if old.is_dir() and not new.exists():
            old.replace(new)
    context = mp.get_context("spawn")
    messages = context.Queue()
    processes = []
    logs = {}
    completed = skipped = 0
    started = last_report = time.monotonic()
    def report(detail):
        print("[progress] AI picking | completed={} skipped={} total={} | {} | elapsed {:.0f}s".format(
            completed, skipped, (end-start).days, detail, time.monotonic()-started), flush=True)
    try:
        for index, (first, stop) in enumerate(blocks, 1):
            interval = first.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d")
            log = str(log_dir / ("pick_" + interval + ".log"))
            logs[index] = log
            kwargs = dict(ai_pal_root=str(Path(ai_pal_root).resolve()),
                picker_specs=picker_specs, data_dir=str(Path(data_dir).resolve()),
                station_file=str(Path(station_file).resolve()), time_range=interval,
                individual_pick_root=str(Path(individual_pick_root).resolve()),
                ensemble_pick_dir=str(Path(ensemble_pick_dir).resolve()),
                num_workers=int(preprocessing_workers), overwrite=bool(overwrite))
            print("AI worker {}: {} | preprocessing workers={} | log {}".format(
                index, interval, preprocessing_workers, log), flush=True)
            process = context.Process(target=_worker, args=(
                index, kwargs, str(Path(config_path).resolve()), log, messages))
            process.start()
            processes.append(process)
        done = set()
        report("workers started")
        while len(done) < len(processes):
            try:
                kind, index, payload = messages.get(timeout=1)
            except queue.Empty:
                for index, process in enumerate(processes, 1):
                    if process.exitcode is not None and index not in done:
                        raise RuntimeError("AI worker {} exited ({}); see {}".format(index, process.exitcode, logs[index]))
            else:
                if kind == "day":
                    day, was_skipped, memory = payload
                    skipped += was_skipped
                    completed += not was_skipped
                    report("worker{}:{} | GPU peaks {}".format(index, day, memory))
                    last_report = time.monotonic()
                elif kind == "done":
                    done.add(index)
                else:
                    raise RuntimeError("AI worker {} failed: {}; see {}".format(index, payload, logs[index]))
            if time.monotonic() - last_report >= 30:
                report("active blocks={}".format(len(processes)-len(done)))
                last_report = time.monotonic()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError("AI worker exited with code {}".format(process.exitcode))
        report("all blocks complete")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        messages.close()
        messages.join_thread()
