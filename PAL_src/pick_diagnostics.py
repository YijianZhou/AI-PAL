"""Opt-in, bounded per-worker timing and CUDA allocator tracing."""
import functools
import json
import os
import subprocess
from pathlib import Path
from threading import Lock, get_ident
import time


def tensor_metadata(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(getattr(value, 'device', 'cpu'))}
    return None


def process_gpu_memory():
    """External process accounting, sampled only at station boundaries."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=5, check=True)
        return {'nvidia_smi': [line.strip() for line in result.stdout.splitlines()
                               if line.split(',')[0].strip() == str(os.getpid())]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'nvidia_smi_error': str(exc)}


def install(runner):
    root = os.environ.get('AI_PAL_DIAGNOSTICS_DIR')
    if not root or getattr(runner, '_diagnostics_installed', False):
        return
    import torch
    runner._diagnostics_installed = True
    duration = float(os.environ.get('AI_PAL_DIAGNOSTICS_SECONDS', '600'))
    synchronize = os.environ.get('AI_PAL_DIAGNOSTICS_SYNC', '0') == '1'
    Path(root).mkdir(parents=True, exist_ok=True)
    path = Path(root) / ('worker_{}_{}.jsonl'.format(os.getpid(), time.time_ns()))
    lock = Lock()
    started = [None]

    def active():
        return started[0] is not None and time.monotonic() - started[0] < duration

    def emit(stage, device=None, **fields):
        record = dict(time=time.time(), pid=os.getpid(), thread_id=get_ident(), stage=stage, **fields)
        if device is not None and str(device).startswith('cuda'):
            stats = torch.cuda.memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            record.update(device=str(device), allocated_bytes=allocated,
                          reserved_bytes=reserved, unused_reserved_bytes=reserved-allocated,
                          lifetime_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          lifetime_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          inactive_split_bytes=stats.get('inactive_split_bytes.all.current', 0))
        with lock, path.open('a', encoding='utf-8') as fp:
            fp.write(json.dumps(record) + '\n')

    def wrap(original, stage, device_of=lambda a: None, identity=lambda a: {}):
        @functools.wraps(original)
        def call(*args, **kwargs):
            if not active():
                return original(*args, **kwargs)
            device = device_of(args)
            fields = identity(args)
            sync = synchronize and device is not None and str(device).startswith('cuda')
            if sync:
                torch.cuda.synchronize(device)
            emit(stage + ':before', device, **fields)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if sync:
                    torch.cuda.synchronize(device)
                emit(stage + ':after', device, seconds=time.perf_counter()-start, **fields)
        return call

    emit('setup', synchronize=synchronize, duration_seconds=duration,
         torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
         note='Peaks are process lifetime. Timings overlap; do not sum as wall time.')
    original_day = runner._pick_one_day

    @functools.wraps(original_day)
    def day(*args, **kwargs):
        if started[0] is None:
            started[0] = time.monotonic()
            emit('first_new_day', date=str(args[0]))
        return original_day(*args, **kwargs)

    runner._pick_one_day = day
    runner._read_current_station_stream = wrap(
        runner._read_current_station_stream, 'read_merge_gain', identity=lambda a: {'station': a[0]})
    cls = runner.PreparedPickerStream
    original_prepare = cls.from_raw_stream
    cls.from_raw_stream = classmethod(lambda cls, *a, **kw:
                                     wrap(original_prepare, 'waveform_preprocess')(*a, **kw))
    cls.tensor_for = wrap(cls.tensor_for, 'waveform_transfer', device_of=lambda a: a[1],
                          identity=lambda a: {'station': a[0].net_sta,
                                               'tensor': tensor_metadata(a[0]._cpu_tensor)})
    cls.release_device = wrap(cls.release_device, 'waveform_release', device_of=lambda a: a[1],
                              identity=lambda a: {'station': a[0].net_sta})
    original_load = runner._load_pickers

    def load(*args, **kwargs):
        pickers = original_load(*args, **kwargs)
        for name, picker in pickers.items():
            emit('model_loaded', picker.device, picker=name,
                 effective_batch_size=picker.pick.__func__.__globals__.get('batch_size'),
                 parameter_bytes=sum(p.numel()*p.element_size() for p in picker.model.parameters()))
            device_of = lambda a, d=picker.device: d
            identity = lambda a, n=name: {'picker': n}
            picker.pick = wrap(picker.pick, 'picker_total', device_of, identity)
            shape_identity = lambda a, n=name: {'picker': n,
                                               'input_tensor': tensor_metadata(a[0]) if a else None}
            picker.model.forward = wrap(picker.model.forward, 'model_forward', device_of, shape_identity)
            for component in ('gru_layer', 'attention'):
                module = getattr(picker.model, component, None)
                if module is not None:
                    module.forward = wrap(module.forward, component, device_of, shape_identity)
        return pickers

    runner._load_pickers = load
    original_group = runner._run_device_group

    class TimedLock:
        def __init__(self, underlying, queue, device, station):
            self.underlying, self.queue = underlying, queue
            self.device, self.station = device, station

        def __enter__(self):
            start = time.perf_counter()
            self.underlying.acquire()
            if active():
                emit('queue_wait', queue=self.queue, seconds=time.perf_counter()-start)

        def __exit__(self, *args):
            try:
                # The inference lock is still held and release_device has run.
                if active() and str(self.device).startswith('cuda'):
                    torch.cuda.synchronize(self.device)
                    emit('station_gpu_idle', self.device, station=self.station,
                         **process_gpu_memory())
                    if os.environ.get('AI_PAL_DIAGNOSTICS_EMPTY_CACHE', '0') == '1':
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                        emit('station_after_empty_cache', self.device, station=self.station,
                             **process_gpu_memory())
            finally:
                self.underlying.release()

    def group(members, prepared, device_lock, *args, **kwargs):
        return original_group(members, prepared,
                              TimedLock(device_lock, ','.join(n for n, p in members),
                                        getattr(members[0][1], 'device', None),
                                        getattr(prepared, 'net_sta', None)), *args, **kwargs)

    runner._run_device_group = group
