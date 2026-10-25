"""Dynamic event repicking; daily merging and range publication remain ordered."""
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
               '_repick_resume_run_key'}
    settings = {k: v for k, v in vars(cfg).items() if k not in ignored}
    return digest({'schema': 1, 'config': _fingerprint_value(settings),
                   'local': _fingerprint_value(kwargs['repicker_local_specs']),
                   'global': _fingerprint_value(kwargs['repicker_global_specs']),
                   'station': file_identity(kwargs['station_file']),
                   'data_dir': str(Path(kwargs['data_dir']).resolve())})


def _worker(index, state, kwargs, log_path, tasks, messages):
    with open(log_path, 'w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            repicker = None
            try:
                import runtime_console
                runtime_console._ORIGINAL_STDOUT = log
                runtime_console._ACTIVE_STREAM = None
                import offline_event_postprocessor as post
                cfg = SimpleNamespace(**state)
                cfg.out_root = str(kwargs['final_root'])
                cfg.repick_num_workers = max(1, int(kwargs['num_workers']))
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
                        geometry = post.load_station_geometry(cfg, kwargs['station_file'], day)
                        initial_events = None
                        current_day = day
                    if repicker is None:
                        repicker = post.EventRepicker(
                            kwargs['ai_pal_root'], cfg, kwargs['repicker_local_specs'],
                            kwargs['station_file'], station_dict=geometry,
                            repicker_global_specs=kwargs['repicker_global_specs'])
                    elif changed_day:
                        repicker.set_station_geometry(geometry)
                    print('[repick] worker={} task={} date={}'.format(index, kind, day), flush=True)
                    if kind == 'event':
                        initial_path, event_index, root, key = payload
                        if initial_events is None:
                            initial_events = post.read_phase_file(initial_path)
                        event = initial_events[event_index]
                        if not loaded:
                            repicker.load_pickers()
                            loaded = True
                        stations = dict(geometry)
                        for selector, metadata in geometry.items():
                            stations.setdefault('.'.join(selector.split('.')[:2]), metadata)
                        context, snapshots = [], []
                        try:
                            Path(root).mkdir(parents=True, exist_ok=True)
                            with post.tempfile.TemporaryDirectory(dir=root) as tmp:
                                phase = Path(tmp) / 'event.dat'
                                catalog = Path(tmp) / 'catalog.dat'
                                repicker.write_events([event], phase, catalog)
                                context, _ = post._load_waveform_context(
                                    repicker, [event], cfg, kwargs['data_dir'], stations,
                                    cfg.repick_num_workers)
                                summary = repicker.process_event(event['time'], phase, catalog, context, {})
                                events = post.read_phase_file(phase)
                                if (getattr(cfg, 'enable_event_waveform_plot', False) or
                                    getattr(cfg, 'save_filtered_event_waveform',
                                            getattr(cfg, 'save_filtered_event_waveforms', False))):
                                    snapshots = repicker.capture_event_waveforms(events, context)
                                save_checkpoint(root, key, events, summary, snapshots)
                        finally:
                            post._release_context(context)
                            post._release_snapshots(snapshots)
                        messages.put(('event', index, day))
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
    """Keep only one outstanding task per worker; prioritize ready daily merges."""
    ready = deque(('finalize', day, (plan['cache'], plan['key']))
                  for day, plan in plans.items() if not plan['done'] and not plan['pending'])
    count = min(count, len(tasks) + len(ready))
    if not count:
        return
    logs = Path(kwargs['final_root']).parent / '_internal' / 'postprocess_AI-PAL' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    context = mp.get_context('spawn')
    messages, jobs = context.Queue(), context.Queue()
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
            process = context.Process(target=worker_target, args=(index, state, kwargs, str(log), jobs, messages))
            process.start()
            processes.append(process)
            print('AI repick worker {} | event queue | log {}'.format(index, log), flush=True)
        while tasks or ready or outstanding:
            while outstanding < count and (ready or tasks):
                jobs.put(ready.popleft() if ready else tasks.popleft())
                outstanding += 1
            try:
                kind, index, payload = messages.get(timeout=.5)
            except queue.Empty:
                pass
            else:
                if kind == 'event':
                    completed += 1
                    outstanding -= 1
                    plan = plans[payload]
                    plan['pending'] -= 1
                    if not plan['pending']:
                        ready.append(('finalize', payload, (plan['cache'], plan['key'])))
                elif kind == 'finalized':
                    outstanding -= 1
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
        for _ in processes:
            jobs.put(None)
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
        jobs.cancel_join_thread()
        jobs.close()
        messages.close()


def run_repick_blocks(cfg, kwargs, num_processes, day_complete_callback=None):
    """Compatibility entry point: scheduling is now event-based, not block-based."""
    count = validate_process_count(num_processes)
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
