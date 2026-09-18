"""Cut negative RUN training samples into raw-waveform NPY shards."""
import os
import argparse
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import UTCDateTime
from dataset_npy import (
    preprocess,
    stream_to_sample, write_split_shards, save_shard_index,
    write_cut_progress,
)
import sys
from pathlib import Path

_PICKER_DIR = Path(__file__).resolve().parents[1]
if str(_PICKER_DIR) not in sys.path:
    sys.path.insert(0, str(_PICKER_DIR))
import config
import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()
dtime2str = cfg.dtime2str
to_prep = cfg.to_prep
samp_rate = cfg.samp_rate
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
win_len = cfg.win_len
win_npts = int(win_len * samp_rate)
read_fpha = cfg.read_fpha
read_fpick = cfg.read_fpick
read_assoc_rate = cfg.read_assoc_rate
load_station_stream = cfg.load_station_stream
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
max_assoc_ratio = cfg.max_assoc_ratio
num_aug = cfg.num_aug
positive_num_aug_mode = getattr(cfg, 'positive_num_aug_mode', 'fixed')
if positive_num_aug_mode == 'fixed' and (num_aug is None or num_aug < 1):
    raise ValueError('set num_aug >= 1 when positive_num_aug_mode is "fixed"')
rarity_augmentation_values = getattr(cfg, 'rarity_augmentation_values', None)
rarity_percentiles = getattr(cfg, 'rarity_percentiles', None)


def phase_is_available(value):
    return float(value) >= 0


def expected_positive_multiplier():
    if positive_num_aug_mode == 'fixed':
        train_multiplier = float(num_aug)
    elif positive_num_aug_mode == 'phase':
        values = np.asarray(rarity_augmentation_values, dtype=float)
        percentiles = np.asarray(rarity_percentiles, dtype=float)
        if len(values) != len(percentiles) + 1:
            raise ValueError('rarity augmentation values must have one more item than percentiles')
        edges = np.concatenate(([0.0], percentiles, [100.0]))
        if np.any(np.diff(edges) < 0) or edges[0] != 0 or edges[-1] != 100:
            raise ValueError('rarity percentiles must be sorted values between 0 and 100')
        weights = np.diff(edges) / 100.0
        train_multiplier = float(np.sum(weights * values))
    else:
        raise ValueError('positive_num_aug_mode must be "phase" or "fixed"')
    return train_ratio * train_multiplier + valid_ratio


def get_pick_dict(event_list):
    pick_dict = {}
    for i, [_, picks] in enumerate(event_list):
      for net_sta, pick in picks.items():
        tp, ts = pick[:2]
        phase_anchor = tp if phase_is_available(tp) else ts
        sta_date = '%s_%s' % (net_sta, phase_anchor.date)
        pick_dict.setdefault(sta_date, []).append([tp, ts])
    return pick_dict


def cut_event_window(day_stream, t0, t1):
    st = day_stream.copy().slice(t0-win_len/2, t1+win_len/2)
    if 0 in st.max() or len(st) != 3:
        return None
    if to_prep:
        st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st) != 3:
        return None
    amax_sec = [np.argmax(abs(tr.data))/samp_rate for tr in st]
    if min(amax_sec) > win_len/2:
        return None
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    for tr in st:
        tr.data[np.isnan(tr.data)] = 0
        tr.data[np.isinf(tr.data)] = 0
    return st


class Negative(Dataset):
  def __init__(self, pick_num_items, pick_dict, cut_neg_ratio, data_dir, out_root, shard_size):
    self.pick_num_items = pick_num_items
    self.pick_dict = pick_dict
    self.cut_neg_ratio = cut_neg_ratio
    self.data_dir = data_dir
    self.out_root = out_root
    self.shard_size = shard_size

  def __getitem__(self, index):
    train_samples, valid_samples = [], []
    sta_date, [num_unassoc, num_assoc] = self.pick_num_items[index]
    net_sta, date = sta_date.split('_')
    date = UTCDateTime(date)
    day_stream = load_station_stream(date, self.data_dir, net_sta)
    if not day_stream or num_unassoc == 0:
        return [], []
    assoc_ratio = num_assoc / (num_unassoc + num_assoc)
    if assoc_ratio >= max_assoc_ratio:
        return [], []
    num_cut = int(
        num_unassoc * self.cut_neg_ratio
        * (max_assoc_ratio - assoc_ratio) / max_assoc_ratio
    )
    dtype = [('tp','O'),('ts','O')]
    picks = self.pick_dict[sta_date] if sta_date in self.pick_dict else []
    picks = np.array([(tp,ts) for tp,ts in picks], dtype=dtype)
    for _ in range(num_cut):
        rand = np.random.rand(1)[0]
        if rand < train_ratio:
            samp_class = 'train'
        elif rand < train_ratio + valid_ratio:
            samp_class = 'valid'
        else:
            continue
        start_time = date + win_len/2 + np.random.rand(1)[0] * (86400-win_len*1.5)
        end_time = start_time + win_len
        is_tp = np.asarray([
            phase_is_available(value) and start_time < value < end_time
            for value in picks['tp']
        ])
        is_ts = np.asarray([
            phase_is_available(value) and start_time < value < end_time
            for value in picks['ts']
        ])
        if np.any(is_tp | is_ts):
            continue
        st = cut_event_window(day_stream, start_time, end_time)
        if not st:
            continue
        sample = stream_to_sample(st, -1.0, -1.0, win_npts)
        if samp_class == 'train':
            train_samples.append(sample)
        if samp_class == 'valid':
            valid_samples.append(sample)
    safe_group = ('%07d_%s' % (index, sta_date)).replace('.', '_').replace(':', '-')
    train_rows = write_split_shards(self.out_root, 'train', 'negative', safe_group, train_samples, self.shard_size)
    valid_rows = write_split_shards(self.out_root, 'valid', 'negative', safe_group, valid_samples, self.shard_size)
    return train_rows, valid_rows

  def __len__(self):
    return len(self.pick_num_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--fassoc_rate', type=str)
    parser.add_argument('--fpick', type=str)  # legacy fallback
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--shard_size', type=int, default=1024)
    args = parser.parse_args()
    event_list, num_pos = read_fpha(args.fpha)
    positive_multiplier = expected_positive_multiplier()
    negative_target_count = int(round(num_pos * positive_multiplier))
    print(
        'negative baseline target = %d from %d original phases x %.4f expected positive multiplier (%s mode)'
        % (negative_target_count, num_pos, positive_multiplier, positive_num_aug_mode),
        flush=True,
    )
    pick_dict = get_pick_dict(event_list)
    if args.fassoc_rate:
        pick_num_dict, num_picks = read_assoc_rate(args.fassoc_rate)
    elif args.fpick:
        pick_num_dict, num_picks = read_fpick(args.fpick, args.fpha)
    else:
        raise ValueError('provide --fassoc_rate (preferred) or --fpick')
    pick_num_items = list(pick_num_dict.items())
    num_unassociated = sum(counts[0] for counts in pick_num_dict.values())
    if num_unassociated <= 0:
        raise ValueError('association-rate input contains no unassociated picks')
    cut_neg_ratio = negative_target_count / num_unassociated
    planned_by_item = []
    for _, [num_unassoc, num_assoc] in pick_num_items:
        if num_unassoc == 0:
            planned_by_item.append(0)
            continue
        assoc_ratio = num_assoc / (num_unassoc + num_assoc)
        if assoc_ratio >= max_assoc_ratio:
            planned_by_item.append(0)
            continue
        planned_by_item.append(int(
            num_unassoc * cut_neg_ratio
            * (max_assoc_ratio - assoc_ratio) / max_assoc_ratio
        ))
    planned_attempts = sum(planned_by_item)
    print(
        'planned negative attempts = %d for %d positive samples'
        % (planned_attempts, negative_target_count),
        flush=True,
    )
    processed_attempts = 0
    generated_samples = 0
    train_rows, valid_rows = [], []
    dataset = Negative(pick_num_items, pick_dict, cut_neg_ratio, args.data_dir, args.out_root, args.shard_size)
    write_cut_progress(
        args.out_root, 'negative', 0, len(dataset), 0,
        planned_attempts, 0,
    )
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i, [train_rows_i, valid_rows_i] in enumerate(dataloader):
        train_rows += train_rows_i
        valid_rows += valid_rows_i
        processed_attempts += planned_by_item[i]
        generated_samples += sum(
            int(row[1]) for row in train_rows_i + valid_rows_i
        )
        if i % 100 == 0:
            print('%s/%s sta-date pairs done/total' % (i + 1, len(dataset)), flush=True)
            write_cut_progress(
                args.out_root, 'negative', i + 1, len(dataset),
                processed_attempts, planned_attempts, generated_samples,
            )
    train_rows.sort(key=lambda item: item[0])
    valid_rows.sort(key=lambda item: item[0])
    save_shard_index(os.path.join(args.out_root, 'train_neg.npy'), train_rows)
    save_shard_index(os.path.join(args.out_root, 'valid_neg.npy'), valid_rows)
    write_cut_progress(
        args.out_root, 'negative', len(dataset), len(dataset),
        processed_attempts, planned_attempts, generated_samples,
        finished=True,
    )
