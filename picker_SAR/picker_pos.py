"""SAR picker for fixed positive-event NPY shard datasets.

This module is for benchmark/event-window inference, not continuous streams.  It
reads already-preprocessed NPY shards with shape (N, 3, win_npts + 2), applies
random 25 s sub-window ensemble picking, and reports predicted phase times
relative to the original NPY event-window start time.
"""
import glob
import hashlib
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .models import SAR
    from . import config
except ImportError:
    from models import SAR
    import config

cfg = config.Config()

samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = cfg.win_len
win_len_npts = int(win_len * samp_rate)
step_len = cfg.rnn_step_len
step_len_npts = int(step_len * samp_rate)
step_stride = cfg.rnn_step_stride
step_stride_npts = int(step_stride * samp_rate)
num_steps = cfg.rnn_num_steps
global_max_norm = cfg.global_max_norm
trig_thres = cfg.trig_thres
tp_dev = cfg.tp_dev
ts_dev = cfg.ts_dev


pos_start_range = None
pos_num_repeat = None
pos_batch_size = None
pos_seed = None
pos_min_cluster_size = None


def configure_positive_picker(start_range, num_repeat, batch_size, random_seed,
                              min_cluster_size):
    """Set benchmark-only random-window controls supplied by the executable."""
    global pos_start_range, pos_num_repeat, pos_batch_size
    global pos_seed, pos_min_cluster_size
    pos_start_range = [float(start_range[0]), float(start_range[1])]
    pos_num_repeat = int(num_repeat)
    pos_batch_size = int(batch_size)
    pos_seed = int(random_seed)
    pos_min_cluster_size = int(min_cluster_size)
    if pos_num_repeat < 1 or pos_batch_size < 1 or pos_min_cluster_size < 1:
        raise ValueError("positive-picker repeat, batch, and cluster sizes must be positive")


class SARPositivePicker(object):
  """SAR picker for fixed event windows saved as NPY shards."""

  def __init__(self, ckpt_dir, ckpt_idx=-1, gpu_idx=0):
    if os.path.isfile(ckpt_dir):
        ckpt_path = ckpt_dir
    else:
        if int(ckpt_idx) == -1:
            best_path = os.path.join(ckpt_dir, 'best.ckpt')
            numbered = glob.glob(os.path.join(ckpt_dir, '[0-9]*_*.ckpt'))
            if os.path.isfile(best_path):
                ckpt_path = best_path
            elif not numbered:
                raise FileNotFoundError('No .ckpt files found in {}'.format(ckpt_dir))
            else:
                ckpt_idx = max(int(os.path.basename(path).split('_')[0]) for path in numbered)
                ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt' % ckpt_idx)))[0]
        else:
            ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt' % ckpt_idx)))[0]
    print('SAR checkpoint: {}'.format(ckpt_path), flush=True)
    self.device = torch.device(
        'cuda:%s' % gpu_idx
        if int(gpu_idx) >= 0 and torch.cuda.is_available() else 'cpu'
    )
    self.model = SAR()
    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
    self.model.to(self.device)
    self.model.eval()

  def pick_shard(self, shard, meta_rows):
    """Pick all event windows in one shard.

    Parameters
    ----------
    shard : ndarray
        Shape (N, 3, win_npts + 2). Columns 0/1 carry reference P/S times;
        waveform starts at column 2.
    meta_rows : list[dict]
        Metadata rows aligned with shard rows.
    """
    if shard.size == 0:
        return []
    data = np.asarray(shard[:, :num_chn, 2:], dtype=np.float32)
    event_npts = data.shape[2]
    if event_npts < win_len_npts:
        return []

    jobs = []
    win_data = []
    for sample_idx in range(data.shape[0]):
        starts = self.random_window_starts(meta_rows[sample_idx], event_npts)
        for start_sec in starts:
            start_idx = int(round(start_sec * samp_rate))
            end_idx = start_idx + win_len_npts
            if start_idx < 0 or end_idx > event_npts:
                continue
            start_sec_actual = start_idx / float(samp_rate)
            win_data.append(data[sample_idx, :, start_idx:end_idx])
            jobs.append((sample_idx, start_sec_actual))

    phase_votes = [[[], []] for _ in range(data.shape[0])]
    if win_data:
        arr = np.stack(win_data, axis=0).astype(np.float32, copy=False)
        for start in range(0, arr.shape[0], pos_batch_size):
            batch = torch.from_numpy(arr[start:start + pos_batch_size]).to(self.device, non_blocking=True).float()
            batch = self.preprocess_cuda_batch(batch)
            data_seq = self.window_batch_to_seq(batch)
            with torch.inference_mode():
                pred_logits = self.model(data_seq)
                pred_probs = F.softmax(pred_logits, dim=-1).detach().cpu().numpy()
            for local_idx, pred_prob in enumerate(pred_probs):
                sample_idx, win_start_sec = jobs[start + local_idx]
                p_votes, s_votes = self.decode_one_window(pred_prob, win_start_sec)
                phase_votes[sample_idx][0].extend(p_votes)
                phase_votes[sample_idx][1].extend(s_votes)

    out_rows = []
    for sample_idx, meta in enumerate(meta_rows):
        out_rows.extend(self.cluster_sample_votes(meta, 'P', phase_votes[sample_idx][0], tp_dev))
        out_rows.extend(self.cluster_sample_votes(meta, 'S', phase_votes[sample_idx][1], ts_dev))
    return out_rows

  def random_window_starts(self, meta, event_npts):
    event_len = event_npts / float(samp_rate)
    low = float(pos_start_range[0])
    high = float(pos_start_range[1])
    high = min(high, event_len - win_len)
    if high < low:
        return []
    key = '{}|{}|{}|{}'.format(
        meta.get('dataset', ''), meta.get('sb_idx', ''), meta.get('trace_name', ''), meta.get('row_in_shard', '')
    )
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    seed = (int(digest[:8], 16) + pos_seed) % (2**32)
    rng = np.random.default_rng(seed)
    if pos_num_repeat <= 1:
        return np.asarray([(low + high) / 2.0], dtype=np.float32)
    return rng.uniform(low, high, size=pos_num_repeat).astype(np.float32)

  def window_batch_to_seq(self, win_batch):
    data_seq = win_batch.unfold(2, step_len_npts, step_stride_npts)
    data_seq = data_seq.permute(0, 2, 1, 3).reshape(win_batch.size(0), -1, num_chn * step_len_npts)
    if data_seq.size(1) < num_steps:
        pad = data_seq.new_zeros(data_seq.size(0), num_steps - data_seq.size(1), data_seq.size(2))
        data_seq = torch.cat((data_seq, pad), dim=1)
    elif data_seq.size(1) > num_steps:
        data_seq = data_seq[:, 0:num_steps, :]
    return data_seq.contiguous()

  def preprocess_cuda_batch(self, data):
    # Re-normalize each sliced 25 s window; the source shard may be normalized over a longer event window.
    data -= torch.mean(data, dim=2, keepdim=True)
    if global_max_norm:
        scale = torch.amax(torch.abs(data), dim=(1, 2), keepdim=True)
    else:
        scale = torch.amax(torch.abs(data), dim=2, keepdim=True)
    return data / scale.clamp_min(1e-12)

  def decode_one_window(self, pred_prob, win_start_sec):
    pred_prob_p = np.asarray(pred_prob[:, 1], dtype=np.float32)
    pred_prob_s = np.asarray(pred_prob[:, 2], dtype=np.float32)
    pred_prob_p[~np.isfinite(pred_prob_p)] = 0.0
    pred_prob_s[~np.isfinite(pred_prob_s)] = 0.0
    return (
        self.decode_phase(pred_prob_p, win_start_sec),
        self.decode_phase(pred_prob_s, win_start_sec),
    )

  def decode_phase(self, prob, win_start_sec):
    idxs = np.where(prob >= trig_thres)[0]
    if len(idxs) == 0:
        return []
    dets = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
    votes = []
    for det in dets:
        if len(det) == 0:
            continue
        idx = float(np.median(det))
        phase_time = float(win_start_sec + step_len / 2.0 + step_stride * idx)
        phase_prob = float(np.amax(prob[det]))
        votes.append((phase_time, phase_prob))
    return votes

  def cluster_sample_votes(self, meta, phase, votes, dev):
    if not votes:
        return []
    votes = sorted(votes, key=lambda item: item[0])
    clusters = []
    current = [votes[0]]
    for vote in votes[1:]:
        if vote[0] - current[-1][0] <= dev:
            current.append(vote)
        else:
            clusters.append(current)
            current = [vote]
    clusters.append(current)

    rows = []
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) < pos_min_cluster_size:
            continue
        times = np.asarray([item[0] for item in cluster], dtype=np.float32)
        probs = np.asarray([item[1] for item in cluster], dtype=np.float32)
        rows.append({
            'dataset': meta.get('dataset', ''),
            'event_id': meta.get('event_id', ''),
            'trace_name': meta.get('trace_name', ''),
            'station_id': station_id(meta),
            'phase': phase,
            'cluster_idx': cluster_idx,
            'pick_time': float(np.median(times)),
            'pick_time_std': float(np.std(times)),
            'pick_prob': float(np.median(probs)),
            'pick_prob_std': float(np.std(probs)),
            'num_votes': int(len(cluster)),
            'sb_idx': meta.get('sb_idx', ''),
            'shard_path': meta.get('shard_path', ''),
            'row_in_shard': meta.get('row_in_shard', ''),
            'window_start_sec': meta.get('window_start_sec', ''),
            'p_ref_sec': meta.get('p_rel_sec', meta.get('p_sec', '')),
            's_ref_sec': meta.get('s_rel_sec', meta.get('s_sec', '')),
        })
    return rows


def station_id(meta):
    sta = str(meta.get('station_key', '') or '')
    channel = str(meta.get('channel', '') or '')
    if not sta:
        sta = str(meta.get('trace_name', '') or '')
    if channel and channel not in sta:
        return '{}.{}'.format(sta, channel)
    return sta


def rows_to_csv(rows, fout, write_header=False):
    columns = [
        'dataset', 'event_id', 'trace_name', 'station_id', 'phase', 'cluster_idx',
        'pick_time', 'pick_time_std', 'pick_prob', 'pick_prob_std', 'num_votes',
        'sb_idx', 'shard_path', 'row_in_shard', 'window_start_sec', 'p_ref_sec', 's_ref_sec'
    ]
    if write_header:
        fout.write(','.join(columns) + '\n')
    for row in rows:
        vals = [format_value(row.get(col, '')) for col in columns]
        fout.write(','.join(vals) + '\n')


def format_value(value):
    if isinstance(value, float):
        if not np.isfinite(value):
            return ''
        return '{:.5f}'.format(value)
    text = str(value)
    if ',' in text:
        text = text.replace(',', ';')
    return text
