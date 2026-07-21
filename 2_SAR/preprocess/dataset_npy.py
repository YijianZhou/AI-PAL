"""NPY shard reader/writer helpers for SAR zarr generation."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import config

cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.rnn_num_steps
win_len = int(cfg.win_len * samp_rate)
step_len = int(cfg.rnn_step_len * samp_rate)
step_stride = int(cfg.rnn_step_stride * samp_rate)

def stream_to_sample(stream, tp_rel=-1.0, ts_rel=-1.0, win_npts=None):
    if win_npts is None:
        win_npts = min(len(tr.data) for tr in stream[:3])
    out = np.zeros((3, int(win_npts) + 2), dtype=np.float32)
    out[:, 0] = float(tp_rel)
    out[:, 1] = float(ts_rel)
    for ii, tr in enumerate(stream[:3]):
        data = np.asarray(tr.data, dtype=np.float32)
        npts = min(data.size, int(win_npts))
        out[ii, 2:2+npts] = data[:npts]
    return out


def write_split_shards(out_root, split, label, group_name, samples, shard_size=1024):
    if not samples:
        return []
    split_dir = Path(out_root) / split / label
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard_idx, start in enumerate(range(0, len(samples), int(shard_size))):
        chunk = samples[start:start + int(shard_size)]
        shard = np.stack(chunk, axis=0).astype(np.float32, copy=False)
        shard_path = split_dir / ('%s_shard_%06d.npy' % (group_name, shard_idx))
        np.save(shard_path, shard)
        rows.append((str(shard_path), str(shard.shape[0])))
    return rows


def save_shard_index(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(rows, dtype=str))

def get_seq_target(tp, ts):
    target_seq = np.zeros(num_steps, dtype=np.int32)
    if tp < 0 or ts < 0:
        return target_seq
    tp_idx, ts_idx = tp*samp_rate, ts*samp_rate
    tp0 = 0 if tp_idx < step_len else int((tp_idx-step_len)/step_stride) + 1
    tp1 = int(tp_idx / step_stride) + 1
    tp0 = max(0, min(tp0, num_steps))
    tp1 = max(0, min(tp1, num_steps))
    target_seq[tp0:tp1] = 1
    ts0 = int((ts_idx-step_len)/step_stride) + 1
    ts1 = int(ts_idx / step_stride) + 1
    if ts0 <= tp0:
        ts0 = tp0 + int((tp1-tp0)/2)
    ts0 = max(0, min(ts0, num_steps))
    ts1 = max(0, min(ts1, num_steps))
    target_seq[ts0:ts1] = 2
    return target_seq


class NpySarShards(Dataset):
    def __init__(self, shard_list):
        self.shard_rows = np.load(shard_list, allow_pickle=False)
        if self.shard_rows.ndim == 1:
            raise ValueError('Expected a shard index with shape (n_shards, 2): %s' % shard_list)
        self.num_samples = int(np.asarray(self.shard_rows[:, 1], dtype=np.int64).sum())

    def __len__(self):
        return self.shard_rows.shape[0]

    def __getitem__(self, index):
        shard_path = str(self.shard_rows[index, 0])
        shard = np.load(shard_path, mmap_mode='r')
        count = int(self.shard_rows[index, 1])
        data = np.asarray(shard[:count, :num_chn, 2:2+win_len], dtype=np.float32)
        tp = np.asarray(shard[:count, 0, 0], dtype=np.float32)
        ts = np.asarray(shard[:count, 0, 1], dtype=np.float32)
        target = np.empty((count, num_steps), dtype=np.int32)
        for ii in range(count):
            target[ii] = get_seq_target(float(tp[ii]), float(ts[ii]))
        return data, target