"""SAC reader for SAR Zarr generation using raw waveform windows."""
import torch
from torch.utils.data import Dataset
from obspy import read, Stream
import numpy as np
import config

cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.rnn_num_steps
win_len = int(cfg.win_len * samp_rate)
step_len = int(cfg.rnn_step_len * samp_rate)
step_stride = int(cfg.rnn_step_stride * samp_rate)


class Sequences(Dataset):
  def __init__(self, sample_list, is_pos):
    self.samples = np.load(sample_list)
    self.is_pos = is_pos

  def __getitem__(self, index):
    st_paths = self.samples[index]
    st = Stream([read(st_path)[0] for st_path in st_paths])
    data = np.zeros((num_chn, win_len), dtype=np.float32)
    for ii, tr in enumerate(st[:num_chn]):
        npts = min(len(tr.data), win_len)
        data[ii, 0:npts] = np.asarray(tr.data[:npts], dtype=np.float32)
    if self.is_pos:
        header = st[0].stats.sac
        tp, ts = header.t0, header.t1
    else:
        tp, ts = -1, -1
    target_seq = get_seq_target(tp, ts, self.is_pos)
    return data, target_seq

  def __len__(self):
    return len(self.samples)


def get_seq_target(tp, ts, is_pos):
    target_seq = np.zeros(num_steps, dtype=np.int32)
    if not is_pos:
        return target_seq
    tp_idx, ts_idx = tp*samp_rate, ts*samp_rate
    tp_step_idx0 = 0 if tp_idx < step_len else int((tp_idx-step_len)/step_stride) + 1
    tp_step_idx1 = int(tp_idx / step_stride) + 1
    tp_step_idx0 = max(0, min(tp_step_idx0, num_steps))
    tp_step_idx1 = max(0, min(tp_step_idx1, num_steps))
    target_seq[tp_step_idx0:tp_step_idx1] = 1
    ts_step_idx0 = int((ts_idx-step_len)/step_stride) + 1
    ts_step_idx1 = int(ts_idx / step_stride) + 1
    if ts_step_idx0 <= tp_step_idx0:
        ts_step_idx0 = tp_step_idx0 + int((tp_step_idx1-tp_step_idx0)/2)
    ts_step_idx0 = max(0, min(ts_step_idx0, num_steps))
    ts_step_idx1 = max(0, min(ts_step_idx1, num_steps))
    target_seq[ts_step_idx0:ts_step_idx1] = 2
    return target_seq