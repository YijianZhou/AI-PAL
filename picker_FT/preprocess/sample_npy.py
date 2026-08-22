"""Raw-waveform NPY shard helpers for independent sample cutting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

_PICKER_DIR = Path(__file__).resolve().parents[1]
if str(_PICKER_DIR) not in sys.path:
    sys.path.insert(0, str(_PICKER_DIR))
import config

_cfg = config.Config()
_DEFAULT_WIN_NPTS = int(_cfg.win_len * _cfg.samp_rate)


def stream_to_sample(stream, tp_rel=-1.0, ts_rel=-1.0, win_npts=None):
    if win_npts is None:
        win_npts = min(len(trace.data) for trace in stream[:3])
    sample = np.zeros((3, int(win_npts) + 2), dtype=np.float32)
    sample[:, 0] = float(tp_rel)
    sample[:, 1] = float(ts_rel)
    for index, trace in enumerate(stream[:3]):
        data = np.asarray(trace.data, dtype=np.float32)
        npts = min(data.size, int(win_npts))
        sample[index, 2:2 + npts] = data[:npts]
    return sample


def write_split_shards(out_root, split, label, group_name, samples,
                       shard_size=1024):
    if not samples:
        return []
    split_dir = Path(out_root) / split / label
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard_index, start in enumerate(range(0, len(samples), int(shard_size))):
        shard = np.stack(samples[start:start + int(shard_size)]).astype(
            np.float32, copy=False
        )
        shard_path = split_dir / (
            "%s_shard_%06d.npy" % (group_name, shard_index)
        )
        np.save(shard_path, shard)
        rows.append((str(shard_path), str(shard.shape[0])))
    return rows


def write_cut_progress(out_root, stage, completed_items, total_items,
                       processed_attempts, planned_attempts,
                       generated_samples, win_npts=None, finished=False):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(out_root)
    if win_npts is None:
        win_npts = _DEFAULT_WIN_NPTS
    payload = {
        "stage": stage,
        "completed_items": int(completed_items),
        "total_items": int(total_items),
        "processed_attempts": int(processed_attempts),
        "planned_attempts": int(planned_attempts),
        "generated_samples": int(generated_samples),
        "bytes_per_sample": int(
            3 * (int(win_npts) + 2) * np.dtype(np.float32).itemsize
        ),
        "finished": bool(finished),
        "disk_total_bytes": int(disk.total),
        "disk_used_bytes": int(disk.used),
        "disk_free_bytes": int(disk.free),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = out_root / ("cut_%s_progress.json" % stage)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def save_shard_index(path, rows):
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    root = index_path.resolve().parent
    portable_rows = []
    for shard_path, count in rows:
        shard_path = Path(shard_path).resolve()
        try:
            stored_path = shard_path.relative_to(root)
        except ValueError:
            stored_path = shard_path
        portable_rows.append((stored_path.as_posix(), str(count)))
    np.save(index_path, np.asarray(portable_rows, dtype=str))
