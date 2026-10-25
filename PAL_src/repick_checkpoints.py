"""Atomic, local-only event checkpoints (never load caches from untrusted sources)."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


class CachedWaveform:
    """Lazy waveform reader; releasing a view must not destroy its checkpoint."""
    def __init__(self, path):
        self.path = str(path)
        self._stream = None

    @property
    def stream(self):
        if self._stream is None:
            with open(self.path, 'rb') as source:
                self._stream = pickle.load(source)
        return self._stream

    def release(self):
        self._stream = None


def valid_checkpoint(root, key):
    try:
        manifest = json.loads((Path(root) / 'complete.json').read_text())
        return (manifest['key'] == key and all(
            (Path(root) / name).is_file() and (Path(root) / name).stat().st_size == size
            for name, size in manifest['files'].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def save_checkpoint(root, key, events, summary, snapshots):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    files = {}

    def write(name, value):
        path = root / name
        with path.open('wb') as target:
            pickle.dump(value, target, protocol=pickle.HIGHEST_PROTOCOL)
            target.flush()
            os.fsync(target.fileno())
        files[name] = path.stat().st_size

    metadata = []
    for index, snapshot in enumerate(snapshots):
        item = dict(snapshot, waveforms={})
        for number, (station, holder) in enumerate(snapshot['waveforms'].items()):
            name = '{}_{}_{}.pkl'.format(generation, index, number)
            write(name, holder.stream)
            item['waveforms'][station] = name
        metadata.append(item)
    payload = generation + '_result.pkl'
    write(payload, (events, summary, metadata))
    partial = root / 'complete.json.partial'
    with partial.open('w') as target:
        json.dump(dict(key=key, payload=payload, files=files), target)
        target.flush()
        os.fsync(target.fileno())
    partial.replace(root / 'complete.json')
    # Previous interrupted generations are never referenced after the commit.
    for path in root.glob('*.pkl'):
        if path.name not in files:
            path.unlink()


def load_checkpoint(root, key):
    root = Path(root)
    if not valid_checkpoint(root, key):
        raise ValueError('Incomplete or stale event checkpoint: {}'.format(root))
    manifest = json.loads((root / 'complete.json').read_text())
    with (root / manifest['payload']).open('rb') as source:
        events, summary, snapshots = pickle.load(source)
    for snapshot in snapshots:
        snapshot['waveforms'] = {station: CachedWaveform(root / name)
                                 for station, name in snapshot['waveforms'].items()}
    return events, summary, snapshots


def release_checkpoint_waveforms(root):
    """After a committed daily status, retain results but remove large windows."""
    root = Path(root)
    manifest = json.loads((root / 'complete.json').read_text())
    for name in manifest['files']:
        if name != manifest['payload']:
            (root / name).unlink(missing_ok=True)
