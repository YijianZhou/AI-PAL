"""Shared pending-event queue with bounded CPU reassociation pipelines."""
import contextlib
from functools import partial
from collections import deque
from datetime import datetime, timedelta
import json
import multiprocessing as mp
from numbers import Integral
from pathlib import Path
import pickle
import queue
import time
import traceback
from types import SimpleNamespace

from repick_checkpoints import (digest, file_identity, valid_checkpoint,
                                save_checkpoint, release_checkpoint_waveforms)
from offline_timing import write_event_timing


def validate_process_count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('repick_num_processes must be a positive integer')
    return int(value)


def _fingerprint_value(value):
    if isinstance(value, partial):
        return {'partial': _fingerprint_value(value.func),
                'args': _fingerprint_value(value.args),
                'keywords': _fingerprint_value(value.keywords)}
    if callable(value):
        name = getattr(value, '__qualname__', None)
        if name is None:
            raise TypeError('Unsupported callable in repick config: {}'.format(type(value).__name__))
        return [getattr(value, '__module__', ''), name]
    if hasattr(value, 'tolist'):
        return _fingerprint_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): _fingerprint_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_fingerprint_value(v) for v in value]
    if isinstance(value, (str, Path)):
        try:
            if Path(value).is_file():
                return file_identity(value)
        except (OSError, ValueError):
            pass
    return value


def _input_identity(path):
    companion = Path(path).with_suffix('.picker_qc.csv')
    return [file_identity(path), file_identity(companion) if companion.is_file() else None]


def _run_key(cfg, kwargs):
    ignored = {'repick_num_processes', 'repick_num_workers', 'progress_interval_sec',
               'out_root', 'event_waveform_complete_callback', 'console_verbosity',
               '_repick_resume_run_key', 'association_num_workers', 'association_table_dir'}
    settings = {k: v for k, v in vars(cfg).items() if k not in ignored}
    return digest({'schema': 1, 'config': _fingerprint_value(settings),
                   'local': _fingerprint_value(kwargs['repicker_local_specs']),
                   'global': _fingerprint_value(kwargs['repicker_global_specs']),
                   'station': file_identity(kwargs['station_file']),
                   'data_dir': str(Path(kwargs['data_dir']).resolve())})


def _process_event_batch(post, repicker, events, payloads, cfg, kwargs, stations):
    """Retain at most one small batch of waveform contexts; checkpoint each event."""
    with contextlib.ExitStack() as stack:
        pending = []
        for initial_path, event_index, root, key in payloads:
            event_started = time.perf_counter()
            timing = {}
            event = events[event_index]
            Path(root).mkdir(parents=True, exist_ok=True)
            tmp = stack.enter_context(post.tempfile.TemporaryDirectory(dir=root))
            phase, catalog = Path(tmp) / 'event.dat', Path(tmp) / 'catalog.dat'
            repicker.write_events([event], phase, catalog)
            context, _ = post._load_waveform_context(
                repicker, [event], cfg, kwargs['data_dir'], stations, cfg.repick_num_workers, timing)
            stack.callback(post._release_context, context)
            steps = repicker._process_hour_steps(event['time'], event['time'], phase,
                                                 catalog, context, {})
            stack.callback(steps.close)
            try:
                next(steps)  # Serial repicking; queues CPU reassociation before yielding.
                summary = None
            except StopIteration as done:
                summary = done.value
            pending.append((steps, summary, phase, context, root, key, timing, event_started, event))
        for steps, summary, phase, context, root, key, timing, event_started, event in pending:
            if summary is None:
                try:
                    next(steps)
                    raise RuntimeError('Unexpected extra repicking continuation')
                except StopIteration as done:
                    summary = done.value
            outputs = post.read_phase_file(phase)
            snapshots = []
            try:
                snapshot_started = time.perf_counter()
                if (getattr(cfg, 'enable_event_waveform_plot', False) or
                        getattr(cfg, 'save_filtered_event_waveform', False)):
                    snapshots = repicker.capture_event_waveforms(outputs, context)
                timing['checkpoint_snapshot_sec'] = time.perf_counter() - snapshot_started
                checkpoint_started = time.perf_counter()
                save_checkpoint(root, key, outputs, summary, snapshots)
                timing['checkpoint_write_sec'] = time.perf_counter() - checkpoint_started
                timing['event_wall_sec'] = time.perf_counter() - event_started
                timing['origin_time'] = str(event['time'])
                for name, value in summary.items():
                    if name.endswith('_sec') and isinstance(value, (int, float)):
                        timing['repick_' + name] = value
                for name, value in summary.get('picker_seconds', {}).items():
                    timing['picker_' + name + '_sec'] = value
                write_event_timing(root, key, timing)
            finally:
                post._release_snapshots(snapshots)


def _worker(index, state, kwargs, log_path, tasks, messages):
    worker_started = time.perf_counter()
    model_setup_sec = 0.0
    geometry_sec = 0.0
    with open(log_path, 'w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            repicker = None
            association_pool = None
            try:
                import runtime_console
                runtime_console._ORIGINAL_STDOUT = log
                runtime_console._ACTIVE_STREAM = None
                import offline_event_postprocessor as post
                cfg = SimpleNamespace(**state)
                cfg.out_root = str(kwargs['final_root'])
                cfg.repick_num_workers = max(1, int(kwargs['num_workers']))
                from realtime_association_pool import RealtimeAssociationPool, validate_workers
                association_workers = validate_workers(getattr(cfg, 'association_num_workers', 4))
                if association_workers > 1:
                    association_pool = RealtimeAssociationPool(
                        association_workers, getattr(cfg, 'association_table_dir', './input'))
                if cfg.event_waveform_complete_callback:
                    cfg.event_waveform_complete_callback = lambda *args: messages.put(('waveform', index, args))
                else:
                    cfg.event_waveform_complete_callback = None
                runtime_console.configure(cfg)
                post.configure_torch_backends(cfg)
                loaded = False
                current_day, initial_events, geometry = None, None, None
                while True:
                    task = tasks.get()
                    if task is None:
                        break
                    kind, day, payload = task
                    changed_day = day != current_day
                    if changed_day:
                        geometry_started = time.perf_counter()
                        geometry = post.load_station_geometry(cfg, kwargs['station_file'], day)
                        geometry_sec += time.perf_counter() - geometry_started
                        initial_events = None
                        current_day = day
                    if repicker is None:
                        model_started = time.perf_counter()
                        repicker = post.EventRepicker(
                            kwargs['ai_pal_root'], cfg, kwargs['repicker_local_specs'],
                            kwargs['station_file'], station_dict=geometry,
                            repicker_global_specs=kwargs['repicker_global_specs'])
                        model_setup_sec += time.perf_counter() - model_started
                    elif changed_day:
                        repicker.set_station_geometry(geometry)
                    repicker.association_pool = association_pool
                    print('[repick] worker={} task={} date={}'.format(index, kind, day), flush=True)
                    if kind in ('event', 'events'):
                        payloads = [payload] if kind == 'event' else payload
                        if initial_events is None:
                            initial_events = post.read_phase_file(payloads[0][0])
                        if not loaded:
                            model_started = time.perf_counter()
                            repicker.load_pickers()
                            model_setup_sec += time.perf_counter() - model_started
                            loaded = True
                            print('[timing] worker={} model_setup_sec={:.3f}'.format(
                                index, model_setup_sec), flush=True)
                        stations = dict(geometry)
                        for selector, metadata in geometry.items():
                            stations.setdefault('.'.join(selector.split('.')[:2]), metadata)
                        _process_event_batch(post, repicker, initial_events, payloads,
                                             cfg, kwargs, stations)
                        messages.put(('events', index, (day, len(payloads))))
                    else:
                        cache, day_key = payload
                        span = day.strftime('%Y%m%d') + '-' + (day + timedelta(days=1)).strftime('%Y%m%d')
                        post.run_offline_event_postprocessing(
                            cfg=cfg, **dict(kwargs, time_range=span, overwrite=True),
                            num_processes=1, _publish_outputs=False,
                            _event_cache=cache, _repicker=repicker, _day_key=day_key,
                            _resume_identity={'run_key': state.get('_repick_resume_run_key'),
                                              'input': _input_identity(Path(kwargs['initial_phase_dir']) / ('phase_{}.dat'.format(day)))},
                            day_complete_callback=lambda *args: messages.put(('day', index, args)))
                        for root, key in cache:
                            release_checkpoint_waveforms(root)
                        messages.put(('finalized', index, day))
            except BaseException as exc:
                traceback.print_exc(file=log)
                messages.put(('error', index, repr(exc)))
                raise
            finally:
                if repicker is not None:
                    repicker.close()
                if association_pool is not None:
                    association_pool.close()
                Path(log_path).with_suffix('.timing.json').write_text(json.dumps(dict(
                    worker=index, model_setup_sec=model_setup_sec,
                    station_geometry_sec=geometry_sec,
                    worker_lifetime_wall_sec=time.perf_counter() - worker_started,
                    note='Worker lifetime includes queue waiting, finalization and shutdown; not additive across workers.',
                ), indent=2), encoding='utf-8')


def _plan(cfg, kwargs):
    import offline_event_postprocessor as post
    start, end = [datetime.strptime(v, '%Y%m%d').date() for v in kwargs['time_range'].split('-')]
    if start >= end:
        raise ValueError('time_range must have start < exclusive end')
    root = Path(kwargs['final_root']).parent / '_internal' / 'postprocess_AI-PAL'
    status_root = root / 'postprocess_status'
    legacy_status = Path(kwargs['final_root']) / 'postprocess_status'
    if legacy_status.is_dir() and not status_root.exists():
        status_root = legacy_status
    run_key = _run_key(cfg, kwargs)
    plans, tasks, skipped, total = {}, deque(), 0, 0
    for offset in range((end-start).days):
        day = start + timedelta(days=offset)
        initial = Path(kwargs['initial_phase_dir']) / ('phase_{}.dat'.format(day))
        input_identity = _input_identity(initial)
        phase = root / 'daily' / initial.name
        legacy_phase = Path(kwargs['final_root']) / initial.name
        if not phase.exists() and legacy_phase.exists():
            phase = legacy_phase
        status_path = status_root / ('{}.json'.format(day))
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError):
            status = {}
        compatible = (status.get('repick_version') == post.EVENT_REPICK_VERSION and
                      status.get('magnitude_qc') == post.magnitude_parameters(cfg) and
                      status.get('final_event_min_both_group_picks') == post.minimum_both_group_picks(cfg) and
                      status.get('final_event_min_quality0_picks') == post.minimum_quality0_picks(cfg))
        resume_identity = {'run_key': run_key, 'input': input_identity}
        if (not kwargs['overwrite'] and phase.is_file() and compatible and
                status.get('resume_identity') == resume_identity and
                isinstance(status.get('num_initial_events'), int) and
                status['num_initial_events'] >= 0):
            total += status['num_initial_events']
            skipped += status['num_initial_events']
            plans[day] = dict(phase=phase, status_path=status_path, status=status,
                              done=True, cache=[], key=status['event_checkpoint_key'],
                              pending=0, run_key=run_key)
            continue
        events = post.read_phase_file(initial)
        total += len(events)
        day_key = digest([run_key, post.EVENT_REPICK_VERSION, events])
        done = (not kwargs['overwrite'] and phase.is_file() and compatible and
                status.get('event_checkpoint_key', day_key) == day_key)
        cache, pending = [], 0
        if done:
            skipped += len(events)
        else:
            for index, event in enumerate(events):
                key = digest([run_key, post.EVENT_REPICK_VERSION, event])
                path = root / 'events' / str(day) / ('{:07d}_{}'.format(index, key[:16]))
                cache.append((str(path), key))
                if not kwargs['overwrite'] and valid_checkpoint(path, key):
                    skipped += 1
                else:
                    tasks.append(('event', day, (str(initial), index, str(path), key)))
                    pending += 1
        plans[day] = dict(phase=phase, status_path=status_path, status=status,
                          done=done, cache=cache, key=day_key, pending=pending, run_key=run_key)
    mismatched = sum(bool(p['status'].get('event_checkpoint_key')) and not p['done']
                     for p in plans.values())
    print('[resume] AI repicking | overwrite={} | reusable days={}/{} | reusable events={} | pending events={} | incompatible/incomplete daily markers={}'.format(
        kwargs['overwrite'], sum(p['done'] for p in plans.values()), len(plans), skipped,
        len(tasks), mismatched), flush=True)
    if mismatched:
        print('[resume] Mismatches may reflect changed inputs/settings or old process-dependent fingerprints; neither is accepted without validation.', flush=True)
    return plans, tasks, skipped, total


def _dispatch(plans, tasks, state, kwargs, count, interval, callback, waveform_callback,
              skipped, total, worker_target=_worker):
    """Assign same-day batches dynamically; daily merges wait for all events."""
    ready = deque(('finalize', day, (plan['cache'], plan['key']))
                  for day, plan in plans.items() if not plan['done'] and not plan['pending'])
    count = min(count, len(tasks) + len(ready))
    if not count:
        return
    logs = Path(kwargs['final_root']).parent / '_internal' / 'postprocess_AI-PAL' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    context = mp.get_context('spawn')
    messages = context.Queue()
    jobs = [context.Queue() for _ in range(count)]
    batch_size = validate_process_count(state.get('association_num_workers', 4))
    busy = set()
    processes, outstanding = [], 0
    completed = 0
    started = last_report = time.monotonic()
    def report():
        print('[progress] AI repicking | events completed={} skipped={} total={} | days finalized={}/{} | active workers={} | elapsed {:.0f}s'.format(
            completed, skipped, total, sum(p['done'] for p in plans.values()), len(plans),
            outstanding, time.monotonic()-started), flush=True)
    try:
        for index in range(1, count+1):
            log = logs / ('repick_{}_worker{}.log'.format(kwargs['time_range'], index))
            process = context.Process(target=worker_target, args=(index, state, kwargs, str(log), jobs[index-1], messages))
            process.start()
            processes.append(process)
            print('AI repick worker {} | shared pending-event queue | reassociation workers={} | log {}'.format(
                index, batch_size, log), flush=True)
        while tasks or ready or outstanding:
            for slot in range(count):
                if slot + 1 in busy:
                    continue
                if ready:
                    task = ready.popleft()
                elif tasks:
                    # Reserve work for other idle workers when the queue is short.
                    idle = count - len(busy)
                    limit = min(batch_size, (len(tasks) + idle - 1) // idle)
                    _, day, payload = tasks.popleft()
                    batch = [payload]
                    while (len(batch) < limit and tasks
                           and tasks[0][1] == day):
                        batch.append(tasks.popleft()[2])
                    task = ('events', day, batch)
                else:
                    continue
                jobs[slot].put(task)
                busy.add(slot + 1)
                outstanding += 1
            try:
                kind, index, payload = messages.get(timeout=.5)
            except queue.Empty:
                pass
            else:
                if kind in ('event', 'events'):
                    day, finished = (payload, 1) if kind == 'event' else payload
                    completed += finished
                    outstanding -= 1
                    busy.remove(index)
                    plan = plans[day]
                    plan['pending'] -= finished
                    if not plan['pending']:
                        ready.append(('finalize', day, (plan['cache'], plan['key'])))
                elif kind == 'finalized':
                    outstanding -= 1
                    busy.remove(index)
                    plans[payload]['done'] = True
                elif kind == 'waveform':
                    waveform_callback(*payload)
                elif kind == 'day':
                    plans[payload[0]]['phase'] = Path(payload[1])
                    if callback:
                        callback(*payload)
                else:
                    raise RuntimeError('Repick worker {} failed: {}; see {}'.format(index, payload, logs))
            for index, process in enumerate(processes, 1):
                if process.exitcode is not None:
                    raise RuntimeError('Repick worker {} exited ({}); see {}'.format(index, process.exitcode, logs))
            if time.monotonic()-last_report >= interval:
                report()
                last_report = time.monotonic()
        for channel in jobs:
            channel.put(None)
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError('Repick worker exited ({})'.format(process.exitcode))
        report()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        for channel in jobs:
            channel.cancel_join_thread()
            channel.close()
        messages.close()


def run_repick_blocks(cfg, kwargs, num_processes, day_complete_callback=None):
    """Dynamically queue pending events in bounded repick/reassociation batches."""
    count = validate_process_count(num_processes)
    from realtime_association_pool import validate_workers
    cfg.association_num_workers = validate_workers(getattr(cfg, 'association_num_workers', 4))
    interval = float(getattr(cfg, 'progress_interval_sec', 60))
    if not 0 < interval < float('inf'):
        raise ValueError('progress_interval_sec must be finite and positive')
    plans, tasks, skipped, total = _plan(cfg, kwargs)
    state = dict(vars(cfg))
    state['_repick_resume_run_key'] = next(iter(plans.values()))['run_key']
    waveform_callback = state.pop('event_waveform_complete_callback', None)
    state['event_waveform_complete_callback'] = waveform_callback is not None
    pickle.dumps((state, kwargs))
    for day, plan in plans.items():
        if plan['done'] and day_complete_callback:
            day_complete_callback(day, plan['phase'], plan['status_path'],
                                  dict(plan['status'], skipped_existing=True))
        if not plan['done']:
            # A failed overwrite must not leave a previous success marker valid.
            if plan['status_path'].exists():
                archived = plan['status_path'].with_name(
                    '{}.previous.{}.json'.format(plan['status_path'].stem, time.time_ns()))
                plan['status_path'].replace(archived)
    _dispatch(plans, tasks, state, kwargs, count, interval, day_complete_callback,
              waveform_callback, skipped, total)
    return [plan['phase'] for plan in plans.values()]
