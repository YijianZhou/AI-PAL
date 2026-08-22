"""NPY shard reader for PhaseNet positive/negative targets."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import sys

_PICKER_DIR = Path(__file__).resolve().parents[1]
if str(_PICKER_DIR) not in sys.path:
    sys.path.insert(0, str(_PICKER_DIR))
import config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = int(cfg.win_len * samp_rate)
label_wid = int(cfg.label_wid * samp_rate)


def get_phase_target(tp, ts):
    target_seq = np.zeros([3, win_len], dtype=np.float32)
    target_seq[0, :] = 1.
    if tp < 0 or ts < 0:
        return target_seq
    half_wid = label_wid // 2
    pick_lab = np.exp(-((np.arange(-half_wid, half_wid + 1))**2) / (2*(label_wid/5)**2)).astype(np.float32)

    def put_pick(phase_idx, pick_time):
        pick_idx = pick_time * samp_rate
        lab0 = int(pick_idx - half_wid)
        lab1 = int(pick_idx + half_wid + 1)
        out0 = max(lab0, 0)
        out1 = min(lab1, win_len)
        if out0 >= out1:
            return
        src0 = out0 - lab0
        src1 = src0 + (out1 - out0)
        target_seq[phase_idx, out0:out1] = pick_lab[src0:src1]

    put_pick(1, tp)
    put_pick(2, ts)
    target_seq[0, :] = np.maximum(0., 1. - target_seq[1, :] - target_seq[2, :])
    return target_seq


class NpyWindowShards(Dataset):
    def __init__(self, shard_list):
        self.shard_root = Path(shard_list).resolve().parent
        self.shard_rows = np.load(shard_list, allow_pickle=False)
        if self.shard_rows.ndim == 1:
            raise ValueError('Expected a shard index with shape (n_shards, 2): %s' % shard_list)
        self.num_samples = int(np.asarray(self.shard_rows[:, 1], dtype=np.int64).sum())

    def __len__(self):
        return self.shard_rows.shape[0]

    def __getitem__(self, index):
        shard_path = Path(str(self.shard_rows[index, 0]))
        if not shard_path.is_absolute():
            shard_path = self.shard_root / shard_path
        shard = np.load(shard_path, mmap_mode='r')
        count = int(self.shard_rows[index, 1])
        data = np.asarray(shard[:count, :num_chn, 2:2+win_len], dtype=np.float32)
        tp = np.asarray(shard[:count, 0, 0], dtype=np.float32)
        ts = np.asarray(shard[:count, 0, 1], dtype=np.float32)
        target = np.empty((count, 3, win_len), dtype=np.float32)
        for ii in range(count):
            target[ii] = get_phase_target(float(tp[ii]), float(ts[ii]))
        return data, target


class NpyWindows(NpyWindowShards):
    """Backward-compatible alias for the shard dataset."""
