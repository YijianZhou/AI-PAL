"""Source helpers to copy local negative Zarr arrays without changing splits."""
import json
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager, closing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _open_copy_worker(stores, relative, staged):
    import zarr
    global _worker_arrays, _worker_output
    _worker_arrays = []
    offset = 0
    for store in stores:
        array = zarr.open(str(Path(store) / relative), mode='r')
        _worker_arrays.append((offset, offset + array.shape[0], array))
        offset += array.shape[0]
    _worker_output = zarr.open(str(staged), mode='r+')


def _copy_chunk_range(bounds):
    import numpy as np
    start, end = bounds
    parts = [array[max(start, lo) - lo:min(end, hi) - lo]
             for lo, hi, array in _worker_arrays if lo < end and hi > start]
    _worker_output[start:end] = parts[0] if len(parts) == 1 else np.concatenate(parts)
    return start, end


def _parallel_ranges(stores, relative, staged, cursor, total, step, workers):
    # Ordered acknowledgments preserve a contiguous resume watermark. Bound
    # in-flight tasks; workers never share a destination storage chunk/shard.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=_open_copy_worker,
                             initargs=(stores, relative, staged)) as pool:
        pending = deque()
        for start in range(cursor, total, step):
            pending.append(pool.submit(_copy_chunk_range, (start, min(start + step, total))))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


@contextmanager
def _transfer_guard(destination):
    """An OS lock is released on exit or process death; never unlink this file."""
    with (destination / '.negative_transfer.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another negative transfer is running') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Another negative transfer is running') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _save_transfer_json(path, value):
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def copy_annual_negatives(source, destination, years, models, batch_rows=128,
                          restart_legacy=False, num_workers=1):
    """Resume a staged annual transfer; never alter CEED positive arrays.

    restart_legacy is an explicit assertion that the old, non-locking copier
    has stopped. It restarts copying, not resumes uncheckpointed legacy data.
    """
    destination = Path(destination).resolve()
    with _transfer_guard(destination):
        return _copy_annual_negatives(source, destination, years, models,
                                      batch_rows, restart_legacy, num_workers)


def _copy_annual_negatives(source, destination, years, models, batch_rows,
                           restart_legacy, num_workers=1):
    """Concatenate annual negatives into an existing positive-only CEED store.

    Run with no concurrent readers/writers. Zarr chunks are decoded in bounded
    batches because year boundaries need not coincide with chunk boundaries.
    """
    import zarr
    import numpy as np

    source, destination = Path(source).resolve(), Path(destination).resolve()
    years = tuple(years)
    models = tuple(models)
    if not years or years != tuple(sorted(set(years))):
        raise ValueError('years must be nonempty, unique, and chronological')
    if not isinstance(batch_rows, int) or batch_rows <= 0:
        raise ValueError('batch_rows must be a positive integer')
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError('num_workers must be a positive integer')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be distinct, non-nested stores')
    stores = [source / ('{}.zarr'.format(year)) for year in years]
    names = array_names(models)
    entries = [(split, name) for split in ('train', 'valid') for name in names]
    manifest = destination / 'local_negative_source.json'
    lock = destination / '.negative_transfer_in_progress'
    stage = destination / '.negative_transfer_stage'
    state_path = destination / '.negative_transfer_state.json'
    identity = {'source': str(source), 'years': list(years),
                'models': sorted(set(models)), 'source_stores': [str(s) for s in stores]}
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if manifest.exists():
        report = json.loads(manifest.read_text())
        if any(report.get(key) != value for key, value in identity.items()):
            raise FileExistsError('Existing negative manifest belongs to another transfer')
        validate_mixed_store(destination, models, allow_transfer=True)
        if state and state.get('identity') == identity:
            lock.unlink(missing_ok=True)
        print('[complete] requested negatives already copied', flush=True)
        return
    if state is not None and state.get('identity') != identity:
        raise ValueError('Resume settings differ from saved transfer; restore source/years/models')
    if state is not None and state.get('phase') not in ('copying', 'publishing'):
        raise ValueError('Invalid transfer checkpoint phase')
    if lock.exists() and state is None and not restart_legacy:
        raise FileExistsError('Legacy transfer has no resume checkpoint. Confirm the old '
                              'process stopped, then set RESTART_LEGACY_TRANSFER = True. '
                              'Old temporary files are retained; copying restarts.')
    if state is None and stage.exists():
        raise FileExistsError('Untracked staging directory: ' + str(stage))
    for store in stores:
        validate_source(store, models)
    totals = {}
    source_metadata = {}
    # Open every source before writing so missing/incompatible arrays fail early.
    for split, name in entries:
        target = destination / split / ('negative_' + name)
        if target.exists() and not (state and state.get('phase') == 'publishing'):
            raise FileExistsError(target)
        positive = zarr.open(str(destination / split / ('positive_' + name)), mode='r')
        positive_count = array_shape(destination / split / 'positive_data')[0]
        if positive_count <= 0 or positive.shape[0] != positive_count:
            raise ValueError('Invalid positive split: ' + split)
        total = 0
        for store in stores:
            array_path = store / split / ('negative_' + name)
            array = zarr.open(str(array_path), mode='r')
            meta = array_path / '.zarray'
            if not meta.exists():
                meta = array_path / 'zarr.json'
            source_metadata[str(array_path)] = json.loads(meta.read_text())
            if array.shape[1:] != positive.shape[1:] or array.dtype != positive.dtype:
                raise ValueError('Incompatible shape/dtype: {}/{}/{}'.format(store, split, name))
            total += array.shape[0]
        totals[split] = total
    if state is None:
        state = {'identity': identity, 'phase': 'copying', 'rows': {},
                 'source_metadata': source_metadata}
        _save_transfer_json(state_path, state)
    elif state.get('source_metadata') != source_metadata:
        raise ValueError('Source array metadata changed since transfer started')
    lock.write_text('Annual negative transfer in progress; do not train.\n')
    if state['phase'] == 'copying':
        stage.mkdir(exist_ok=True)
        for split, name in entries:
            relative = Path(split) / ('negative_' + name)
            template = stores[0] / relative
            staged = stage / relative
            staged.mkdir(parents=True, exist_ok=True)
            # Preserve the source array's format, codecs, chunks, and dtype.
            metadata_name = '.zarray' if (template / '.zarray').exists() else 'zarr.json'
            metadata = json.loads((template / metadata_name).read_text(encoding='utf-8'))
            metadata['shape'][0] = totals[split]
            if not (staged / metadata_name).exists():
                _save_transfer_json(staged / metadata_name, metadata)
            if (template / '.zattrs').exists():
                shutil.copyfile(template / '.zattrs', staged / '.zattrs')
            output = zarr.open(str(staged), mode='r+')
            if tuple(output.shape) != tuple(metadata['shape']):
                raise ValueError('Staged shape changed: ' + str(staged))
            key = relative.as_posix()
            completed = state['rows'].get(key, 0)
            if completed == totals[split]:
                continue
            # Replay the boundary chunk: an interrupted write may have touched
            # earlier rows of that chunk (including across a year boundary).
            chunk_rows = (metadata['chunks'][0] if metadata_name == '.zarray'
                          else metadata['chunk_grid']['configuration']['chunk_shape'][0])
            cursor = completed // chunk_rows * chunk_rows
            print('[resume] {} rows={}/{}'.format(key, cursor, totals[split]), flush=True)
            arrays = []
            offset = 0
            for year, store in zip(years, stores):
                array = zarr.open(str(store / relative), mode='r')
                count = array.shape[0]
                arrays.append((offset, offset + count, array))
                offset += count
            # Write whole storage chunks (shards for sharded Zarr). This avoids
            # decoding an incomplete chunk left by a terminated write.
            step = max(chunk_rows, batch_rows // chunk_rows * chunk_rows)
            print('[copy] {} workers={} rows/task={}'.format(key, num_workers, step), flush=True)
            if num_workers > 1:
                completed_ranges = _parallel_ranges(stores, relative, staged, cursor,
                                                    totals[split], step, num_workers)
            else:
                completed_ranges = ((start, min(start + step, totals[split]))
                                    for start in range(cursor, totals[split], step))
            # Join workers before releasing the transfer lock, even if saving
            # a checkpoint fails or the user interrupts the parent process.
            with closing(completed_ranges):
                for start, end in completed_ranges:
                    if num_workers == 1:
                        parts = [array[max(start, lo) - lo:min(end, hi) - lo]
                                 for lo, hi, array in arrays if lo < end and hi > start]
                        output[start:end] = parts[0] if len(parts) == 1 else np.concatenate(parts)
                    state['rows'][key] = end
                    _save_transfer_json(state_path, state)
                    if start == cursor or end == totals[split] or (start // step) % 100 == 0:
                        print('[copy] {} rows={}/{}'.format(key, end, totals[split]), flush=True)
            del output
        state['phase'] = 'publishing'
        _save_transfer_json(state_path, state)
    if state['phase'] == 'publishing':
        for split, name in entries:
            relative = Path(split) / ('negative_' + name)
            target = destination / relative
            if target.exists():
                if (stage / relative).exists():
                    raise FileExistsError(target)
                continue
            (stage / relative).rename(target)
        # Validate while holding the transfer lock (not through trainer guard).
        validate_mixed_store(destination, models, allow_transfer=True)
        report = dict(identity, negative_counts=totals)
        _save_transfer_json(manifest, report)
    lock.unlink()
    print('[complete] negatives={} destination={}'.format(totals, destination), flush=True)


def array_shape(path):
    path = Path(path)
    metadata = path / '.zarray'
    if not metadata.exists():
        metadata = path / 'zarr.json'
    return tuple(json.loads(metadata.read_text(encoding='utf-8'))['shape'])


def array_names(models):
    models = set(models)
    if not models or models - {'SAR', 'FT', 'PHN', 'RUN'}:
        raise ValueError('Select one or more supported models')
    names = ['data']
    if models & {'SAR', 'FT'}:
        names.append('target_frame')
    if models & {'PHN', 'RUN'}:
        names.append('target_sample')
    return names


def validate_source(root, models):
    root = Path(root)
    for split in ('train', 'valid'):
        count = array_shape(root / split / 'negative_data')[0]
        if count <= 0:
            raise ValueError('{} has no {} negatives'.format(root, split))
        for name in array_names(models):
            if array_shape(root / split / ('negative_' + name))[0] != count:
                raise ValueError('Negative data/target count mismatch: ' + split)


def validate_mixed_store(root, models, allow_transfer=False):
    root = Path(root)
    if not allow_transfer and (root / '.negative_transfer_in_progress').exists():
        raise RuntimeError('Negative transfer is incomplete; do not train: ' + str(root))
    validate_source(root, models)
    for split in ('train', 'valid'):
        count = array_shape(root / split / 'positive_data')[0]
        if count <= 0:
            raise ValueError('Empty positive split: ' + split)
        for name in array_names(models):
            pos = array_shape(root / split / ('positive_' + name))
            neg = array_shape(root / split / ('negative_' + name))
            if pos[0] != count or pos[1:] != neg[1:]:
                raise ValueError('Positive/negative array mismatch: {}/{}'.format(split, name))


def copy_local_negatives(source, destination, models):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be distinct, non-nested stores')
    validate_source(source, models)
    entries = [(split, 'negative_' + name)
               for split in ('train', 'valid') for name in array_names(models)]
    for split, name in entries:
        target = destination / split / name
        if target.exists():
            raise FileExistsError(target)
        positive = destination / split / name.replace('negative_', 'positive_', 1)
        if array_shape(positive)[1:] != array_shape(source / split / name)[1:]:
            raise ValueError('Incompatible CEED/local array shapes: ' + str(positive))
    published = []
    # Stage all chunks before publishing; failed transfers leave positives intact.
    with tempfile.TemporaryDirectory(prefix='ceed-neg-', dir=destination.parent) as temp:
        stage = Path(temp)
        for split, name in entries:
            shutil.copytree(source / split / name, stage / split / name)
        try:
            for split, name in entries:
                target = destination / split / name
                (stage / split / name).rename(target)
                published.append(target)
            validate_mixed_store(destination, models)
            report = {'source': str(source), 'models': list(models), 'negative_counts': {
                split: array_shape(destination / split / 'negative_data')[0]
                for split in ('train', 'valid')}}
            (destination / 'local_negative_source.json').write_text(
                json.dumps(report, indent=2) + '\n', encoding='utf-8')
        except Exception:
            for target in published:
                shutil.rmtree(target)
            raise
    print('Copied local negatives: {}'.format(report['negative_counts']), flush=True)
