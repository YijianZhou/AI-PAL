"""NPY shard reader for Frame Transformer positive/negative targets."""
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
num_steps = int((cfg.win_len - cfg.ft_frame_length) / cfg.ft_frame_step) + 1
win_len = int(cfg.win_len * samp_rate)
step_len = int(cfg.ft_frame_length * samp_rate)
step_stride = int(cfg.ft_frame_step * samp_rate)


def get_frame_target(tp, ts):
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


class NpyFrameShards(Dataset):
    def __init__(self, shard_list, include_data=True):
        self.shard_root = Path(shard_list).resolve().parent
        self.shard_rows = np.load(shard_list, allow_pickle=False)
        if self.shard_rows.ndim == 1:
            raise ValueError('Expected a shard index with shape (n_shards, 2): %s' % shard_list)
        self.include_data = include_data
        self.num_samples = int(np.asarray(self.shard_rows[:, 1], dtype=np.int64).sum())

    def __len__(self):
        return self.shard_rows.shape[0]

    def __getitem__(self, index):
        shard_path = Path(str(self.shard_rows[index, 0]))
        if not shard_path.is_absolute():
            shard_path = self.shard_root / shard_path
        shard = np.load(shard_path, mmap_mode='r')
        count = int(self.shard_rows[index, 1])
        tp = np.asarray(shard[:count, 0, 0], dtype=np.float32)
        ts = np.asarray(shard[:count, 0, 1], dtype=np.float32)
        target = np.empty((count, num_steps), dtype=np.int32)
        for ii in range(count):
            target[ii] = get_frame_target(float(tp[ii]), float(ts[ii]))
        if not self.include_data:
            return None, target
        data = np.asarray(shard[:count, :num_chn, 2:2+win_len], dtype=np.float32)
        return data, target


class NpyFrameSequenceShards(NpyFrameShards):
    """Backward-compatible alias; data are raw windows, not pre-unfolded sequences."""


class NpyFrameSequences(NpyFrameShards):
    """Backward-compatible alias; data are raw windows, not pre-unfolded sequences."""
