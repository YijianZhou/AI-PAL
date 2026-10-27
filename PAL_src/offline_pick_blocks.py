"""Dynamically schedule pending dates across persistent AI picking workers."""
import contextlib
from datetime import datetime
import importlib.util
import multiprocessing as mp
import os
import math
from pathlib import Path
from numbers import Integral
import queue
import sys
import time
import traceback



def load_workflow_config(path):
    """Bind a process-private config without rewriting installed source files."""
    spec = importlib.util.spec_from_file_location("config_ai_pal", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["config_ai_pal"] = module
    spec.loader.exec_module(module)
    return module.Config()


def _worker(index, kwargs, config_path, log_path, messages, tasks):
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                cpu_threads = os.environ.get('AI_PAL_CPU_THREADS')
                if cpu_threads is not None:
                    cpu_threads = int(cpu_threads)
                    if cpu_threads < 1:
                        raise ValueError('AI_PAL_CPU_THREADS must be positive')
                    import torch
                    torch.set_num_threads(cpu_threads)
                    torch.set_num_interop_threads(1)
                    print('[startup] CPU intra-op threads={} | inter-op threads=1'.format(
                        cpu_threads), file=log, flush=True)
                cfg = load_workflow_config(config_path)
                from offline_picker_runner import run_offline_picker_ensemble
                import offline_picker_runner
                from pick_diagnostics import install
                install(offline_picker_runner)
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
                    print("[memory] GPU memory peaks (PyTorch only, process lifetime) "
                          "day={}: {}".format(day.date, memory), file=log, flush=True)
                    messages.put(("day", index, (str(day.date), skipped, memory)))
                def assigned_dates():
                    while True:
                        day = tasks.get()
                        if day is None:
                            return
                        messages.put(('active', index, day))
                        yield day
                run_offline_picker_ensemble(cfg=cfg, day_complete_callback=complete,
                                            _date_tasks=assigned_dates(), **kwargs)
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
                            overwrite=False, progress_interval_sec=60):
    progress_interval_sec = float(progress_interval_sec)
    if not math.isfinite(progress_interval_sec) or progress_interval_sec <= 0:
        raise ValueError("progress_interval_sec must be finite and positive")
    for name, value in [('num_workers', num_workers), ('preprocessing_workers', preprocessing_workers)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError('{} must be a positive integer'.format(name))
    start, end = [datetime.strptime(value, "%Y%m%d").date()
                  for value in time_range.split("-")]
    if start >= end:
        raise ValueError("time_range must have start < exclusive end")
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
    cfg = load_workflow_config(config_path)
    from offline_picker_runner import pending_pick_dates
    pending, skipped = pending_pick_dates(cfg, picker_specs, time_range,
        individual_pick_root, ensemble_pick_dir, overwrite)
    print('[resume] AI picking | pending dates={} skipped={} total={}'.format(
        len(pending), skipped, (end-start).days), flush=True)
    if not pending:
        print('[progress] AI picking | completed=0 skipped={} total={} | all dates complete'.format(
            skipped, (end-start).days), flush=True)
        return
    worker_count = min(int(num_workers), len(pending))
    context = mp.get_context("spawn")
    messages = context.Queue()
    tasks = context.Queue()
    processes = []
    logs = {}
    completed = 0
    active = {}
    started = last_report = time.monotonic()
    def report(detail):
        print("[progress] AI picking | completed={} skipped={} total={} | {} | elapsed {:.0f}s".format(
            completed, skipped, (end-start).days, detail, time.monotonic()-started), flush=True)
    try:
        for day in pending:
            tasks.put(day)
        for _ in range(worker_count):
            tasks.put(None)
        for index in range(1, worker_count + 1):
            log = str(log_dir / ('pick_{}_worker{}.log'.format(time_range, index)))
            logs[index] = log
            kwargs = dict(ai_pal_root=str(Path(ai_pal_root).resolve()),
                picker_specs=picker_specs, data_dir=str(Path(data_dir).resolve()),
                station_file=str(Path(station_file).resolve()), time_range=time_range,
                individual_pick_root=str(Path(individual_pick_root).resolve()),
                ensemble_pick_dir=str(Path(ensemble_pick_dir).resolve()),
                num_workers=int(preprocessing_workers), overwrite=bool(overwrite))
            print("AI worker {}: pending-date queue | preprocessing workers={} | log {}".format(
                index, preprocessing_workers, log), flush=True)
            process = context.Process(target=_worker, args=(
                index, kwargs, str(Path(config_path).resolve()), log, messages, tasks))
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
                    active.pop(index, None)
                    day, was_skipped, memory = payload
                    skipped += was_skipped
                    completed += not was_skipped
                    if time.monotonic() - last_report >= progress_interval_sec:
                        report("worker{}:{} | GPU peaks {}".format(index, day, memory))
                        last_report = time.monotonic()
                elif kind == "done":
                    done.add(index)
                    active.pop(index, None)
                elif kind == 'active':
                    active[index] = payload
                else:
                    raise RuntimeError("AI worker {} failed: {}; see {}".format(index, payload, logs[index]))
            if time.monotonic() - last_report >= progress_interval_sec:
                report("active workers={} | dates={}".format(len(active),
                    ','.join('worker{}:{}'.format(i, day) for i, day in sorted(active.items())) or '-'))
                last_report = time.monotonic()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError("AI worker exited with code {}".format(process.exitcode))
        report("all dates complete")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        messages.close()
        messages.join_thread()
        tasks.cancel_join_thread()
        tasks.close()
