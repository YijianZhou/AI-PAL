"""Bounded CPU process scheduling for independent realtime PAL pick clusters."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import pickle
import os
import threading
import time
import tempfile
import weakref
import io
import math
import shutil
import errno
from numbers import Integral

import numpy as np
from association_tables import _attach_tables, _write_table_file


def validate_workers(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('association_num_workers must be a positive integer')
    return int(value)


def independent_clusters(picks, radius):
    """No strict origin-time neighborhood can cross a gap >= ot_dev.

    Keep transitive chains intact: splitting those arbitrarily would change
    the greedy association's pick consumption and failed-hypothesis cache.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('ot_dev must be finite and positive for parallel association')
    if not len(picks):
        return []
    ordered = np.sort(picks, order='sta_ot')
    cuts = [i for i in range(1, len(ordered))
            if ordered['sta_ot'][i] - ordered['sta_ot'][i - 1] >= radius]
    return np.split(ordered, cuts)


_worker_key = None
_worker_engine = None


def _watch_parent():
    """Do not orphan nested CPU workers if an outer model process is killed."""
    import faulthandler
    faulthandler.enable(all_threads=True)
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    def watch():
        while parent.is_alive():
            time.sleep(1)
        os._exit(1)
    threading.Thread(target=watch, daemon=True, name='association-parent-watch').start()




def _associate_cluster(key, descriptor, picks):
    global _worker_key, _worker_engine
    from associator_pal import PS_Pair_Assoc
    # Only small engine metadata changes; table pages are shared read-only.
    if _worker_engine is None or _worker_key != key:
        _worker_engine = None
        _worker_engine = PS_Pair_Assoc.__new__(PS_Pair_Assoc)
        _worker_engine.__dict__.update(descriptor['state'])
        _attach_tables(_worker_engine, descriptor)
        _worker_key = key
    return _worker_engine.associate(picks, verbose=False)




class RealtimeAssociationPool:
    def __init__(self, num_workers, table_dir='./input'):
        self.num_workers = validate_workers(num_workers)
        self._table_lock = threading.Lock()
        self._tables = {}
        self._parents = weakref.WeakSet()
        if not isinstance(table_dir, (str, os.PathLike)) or not os.fspath(table_dir).strip():
            raise ValueError('association_table_dir must be a nonempty directory path')
        table_dir = os.path.abspath(os.path.expanduser(os.fspath(table_dir)))
        os.makedirs(table_dir, exist_ok=True)
        self._table_directory = tempfile.TemporaryDirectory(prefix='ai_pal_assoc_', dir=table_dir)
        # Never fork the realtime parent's CUDA context or running threads.
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_watch_parent,
        )

    def register(self, stations, params, engine=None):
        """Publish one validated engine's tables; workers never construct them.

        association_table_dir selects the backing directory (default ./input). Descriptors
        carry paths/metadata only, never array payloads through process queues.
        """
        from associator_pal import PS_Pair_Assoc
        table_file = params.get('travel_time_table')
        stamp = None
        if isinstance(table_file, (str, os.PathLike)):
            stat = os.stat(table_file)
            stamp = (os.path.realpath(table_file), stat.st_size, stat.st_mtime_ns)
        signature = hashlib.sha256(pickle.dumps(('float32-v1', stations, params, stamp))).hexdigest()
        with self._table_lock:
            descriptor = self._tables.get(signature)
            parent_engine = engine
            if descriptor is None:
                if engine is None or hasattr(engine, '_shared_table_path'):
                    engine = PS_Pair_Assoc(stations, **params)
                names = tuple(stations)
                path = os.path.join(self._table_directory.name, signature + '.npy')
                _write_table_file(path, engine, names)
                excluded = {'tt_dict', 'tt_s_dict', '_station_tables', '_table_reader',
                            '_shared_table_array', '_shared_table_path'}
                state = {k: v for k, v in engine.__dict__.items() if k not in excluded}
                state = pickle.loads(pickle.dumps(state))
                descriptor = dict(path=path, stations=names, state=state)
                self._tables[signature] = descriptor
            if parent_engine is not None:
                engine = parent_engine
                engine.__dict__.update(descriptor['state'])
            if engine is not None and getattr(engine, '_shared_table_path', None) != descriptor['path']:
                _attach_tables(engine, descriptor)
                self._parents.add(engine)
        return signature, descriptor

    def associate(self, key, stations, params, picks):
        clusters = independent_clusters(picks, params['ot_dev'])
        unique = params.get('unique_stations', True)
        clusters = [rows for rows in clusters if
                    (len(set(rows['net_sta'])) if unique else len(rows)) >= params['min_sta']]
        if not clusters:
            return [], [], 0
        signature, descriptor = self.register(stations, params)
        locations, selected = [], []
        futures = [self.executor.submit(_associate_cluster, signature, descriptor, rows)
                   for rows in clusters]
        for future in futures:
            result = future.result()  # Propagate failures; never drop picks.
            if result:
                locations.extend(result[0])
                selected.extend(result[1])
        return locations, selected, len(clusters)

    def submit_event(self, stations, params, picks):
        """One initial event per CPU task; do not split or combine its repicks."""
        # Geometry can change with station epochs. Never reuse a stale table.
        signature, descriptor = self.register(stations, params)
        return self.executor.submit(
            _associate_cluster, signature, descriptor, picks,
        )

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for engine in self._parents:
            engine.tt_dict = {}
            engine.tt_s_dict = {}
            engine._shared_table_array._mmap.close()
            del engine._shared_table_array
            del engine._shared_table_path
        self._parents.clear()
        self._tables.clear()
        self._table_directory.cleanup()
