"""Station-day checkpoints for training cutters (one writer per output root)."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    partial.replace(path)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_rows(root, rows):
    try:
        for name, count in rows:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                return False
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            valid = (array.ndim == 3 and array.shape[0] == int(count)
                     and array.shape[1] == 3 and array.dtype == np.float32)
            del array
            if not valid:
                return False
        return True
    except (OSError, ValueError, TypeError, EOFError):
        return False


def prepare_resume(args, cfg, stage):
    """Bind inputs/settings and seed the randomized positive station-day plan."""
    root = Path(args.out_root).resolve()
    sources = {}
    for key in ('fpha', 'fassoc_rate', 'fpick'):
        value = getattr(args, key, None)
        if value:
            sources[key] = _digest(value)
    # Include inherited config source: it defines waveform loading and controls.
    for cls in type(cfg).__mro__:
        if cls is not object:
            path = inspect.getsourcefile(cls)
            if path:
                sources[cls.__module__] = _digest(path)
    signature = dict(version=1, stage=stage, sources=sources,
                     data_dir=str(Path(args.data_dir).resolve()),
                     shard_size=args.shard_size)
    path = root / '.cut_resume' / stage / 'run.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError('Cutting inputs/config changed; use a new out_root: %s' % root)
    else:
        manifest = dict(signature=signature,
                        seed=int(np.random.randint(0, 2**32, dtype=np.uint32)))
        progress = root / ('cut_%s_progress.json' % stage)
        if progress.exists() and json.loads(progress.read_text()).get('finished'):
            suffix = 'pos' if stage == 'positive' else 'neg'
            rows = []
            for split in ('train', 'valid'):
                index = np.load(root / ('%s_%s.npy' % (split, suffix)), allow_pickle=False)
                split_rows = index.tolist()
                if not _valid_rows(root, split_rows):
                    raise ValueError('Invalid completed legacy shards: %s' % progress)
                rows.append(split_rows)
            manifest['legacy_rows'] = rows
            print('[resume] Adopting completed legacy %s output; previous input/config '
                  'identity was not recorded. Current inputs must match the original run.' % stage,
                  flush=True)
        _atomic_json(path, manifest)
    if 'legacy_rows' in manifest:
        if not all(_valid_rows(root, rows) for rows in manifest['legacy_rows']):
            raise ValueError('Completed legacy output is damaged; use a new out_root')
        print('[resume] %s already complete; skipping waveform cutting' % stage, flush=True)
        return None
    np.random.seed(manifest['seed'])
    return manifest['seed']


class ResumableCut:
    """Publish completion only after both split shard lists have been written."""
    def __init__(self, dataset, stage, seed):
        self.dataset = dataset
        self.stage = stage
        self.seed = seed
        self.root = Path(dataset.out_root).resolve()
        self.items = (dataset.sta_date_items if stage == 'positive'
                      else dataset.pick_num_items)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key = self.items[index][0]
        path = self.root / '.cut_resume' / self.stage / ('%07d.json' % index)
        if path.exists():
            try:
                checkpoint = json.loads(path.read_text(encoding='utf-8'))
                rows = checkpoint['rows']
                if (checkpoint['key'] == key and len(rows) == 2
                        and all(_valid_rows(self.root, split) for split in rows)):
                    return [[(str(self.root / name), count) for name, count in split]
                            for split in rows]
            except (ValueError, KeyError, TypeError):
                pass
            path.unlink()
        # Stable sampling even when worker count or the set of skipped items changes.
        np.random.seed((self.seed + index + 1) % 2**32)
        rows = self.dataset[index]
        relative = [[(str(Path(name).resolve().relative_to(self.root)), count)
                     for name, count in split] for split in rows]
        _atomic_json(path, dict(key=key, rows=relative))
        return rows
