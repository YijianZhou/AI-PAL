"""Cut positive PHN training samples into raw-waveform NPY shards."""
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
samp_rate = cfg.samp_rate
win_len = cfg.win_len
win_npts = int(win_len * samp_rate)
step_len = cfg.p_context_sec
rand_dt_max = win_len/2 - step_len
read_fpha = cfg.read_fpha
load_station_stream = cfg.load_station_stream
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
freq_band = cfg.freq_band
to_prep = cfg.to_prep
global_max_norm = cfg.global_max_norm
num_aug = cfg.num_aug
positive_num_aug_mode = getattr(cfg, 'positive_num_aug_mode', 'fixed')
max_noise = cfg.max_noise


def get_sta_date(event_list):
    sta_date_dict = {}
    for [event_loc, picks] in event_list:
        ot, lat, lon = event_loc[0:3]
        event_name = dtime2str(ot)
        for net_sta, pick in picks.items():
            tp, ts = pick[:2]
            metadata = pick[2] if len(pick) > 2 else {}
            if positive_num_aug_mode == 'phase':
                if 'num_aug' not in metadata:
                    raise ValueError(
                        'phase augmentation mode requires num_aug metadata for '
                        '%s at %s; run 0_analyze_phase_rarity first'
                        % (net_sta, tp)
                    )
                pick_num_aug = int(metadata['num_aug'])
            elif positive_num_aug_mode == 'fixed':
                pick_num_aug = int(num_aug)
            else:
                raise ValueError(
                    'positive_num_aug_mode must be "phase" or "fixed"'
                )
            if pick_num_aug < 1:
                raise ValueError('num_aug must be >= 1 for %s at %s' % (net_sta, tp))
            rand = np.random.rand(1)[0]
            if rand < train_ratio:
                samp_class = 'train'
            elif rand < train_ratio + valid_ratio:
                samp_class = 'valid'
            else:
                continue
            sta_date = '%s_%s' % (net_sta, tp.date)
            sta_date_dict.setdefault(sta_date, []).append(
                [samp_class, event_name, tp, ts, pick_num_aug]
            )
    return sta_date_dict


def add_noise(st, day_stream, tp, ts, picks):
    date = UTCDateTime(st[0].stats.starttime.date)
    t0 = date + win_len/2 + np.random.rand(1)[0] * (86400-win_len*1.5)
    t1 = t0 + win_len
    is_tp = (picks['tp'] > t0) * (picks['tp'] < t1)
    is_ts = (picks['ts'] > t0) * (picks['ts'] < t1)
    if sum(is_tp*is_ts) > 0:
        return st
    st_noise = day_stream.copy().slice(t0-win_len/2, t1+win_len/2)
    if len(st_noise) != 3:
        return st
    if to_prep:
        st_noise = preprocess(st_noise, samp_rate, freq_band)
    st_noise = st_noise.slice(t0, t1).normalize(global_max=global_max_norm)
    if len(st_noise) != 3:
        return st
    npts = min([len(tr) for tr in st + st_noise])
    noise_scale = max_noise * np.random.rand(1)[0]
    for ii in range(3):
        scale = noise_scale * np.amax(abs(st[ii].slice(tp, ts).data))
        st[ii].data[0:npts] += st_noise[ii].data[0:npts] * scale
    return st.detrend('demean').normalize(global_max=global_max_norm)


def cut_event_window(day_stream, t0, t1):
    st = day_stream.copy().slice(t0-win_len/2, t1+win_len/2)
    if 0 in st.max() or len(st) != 3:
        return None
    if to_prep:
        st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st) != 3:
        return None
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    for tr in st:
        tr.data[np.isnan(tr.data)] = 0
        tr.data[np.isinf(tr.data)] = 0
    return st


class Positive(Dataset):
  def __init__(self, sta_date_items, data_dir, out_root, shard_size):
    self.sta_date_items = sta_date_items
    self.data_dir = data_dir
    self.out_root = out_root
    self.shard_size = shard_size

  def __getitem__(self, index):
    train_samples, valid_samples = [], []
    sta_date, samples = self.sta_date_items[index]
    net_sta, date = sta_date.split('_')
    day_stream = load_station_stream(UTCDateTime(date), self.data_dir, net_sta)
    if not day_stream:
        return [], []
    dtype = [('tp','O'),('ts','O')]
    picks = np.array([(tp,ts) for _,_,tp,ts,_ in samples], dtype=dtype)
    for [samp_class, event_name, tp, ts, pick_num_aug] in samples:
        if tp > ts:
            continue
        n_aug = pick_num_aug if samp_class == 'train' else 1
        for aug_idx in range(n_aug):
            rand_dt = min(rand_dt_max, win_len-step_len-(ts-tp))
            start_time = tp - step_len - np.random.rand(1)[0] * rand_dt
            end_time = start_time + win_len
            st = cut_event_window(day_stream, start_time, end_time)
            if not st:
                continue
            if aug_idx > 0 and max_noise > 0:
                st = add_noise(st, day_stream, tp, ts, picks)
            sample = stream_to_sample(st, tp-start_time, ts-start_time, win_npts)
            if samp_class == 'train':
                train_samples.append(sample)
            if samp_class == 'valid':
                valid_samples.append(sample)
    safe_group = ('%07d_%s' % (index, sta_date)).replace('.', '_').replace(':', '-')
    train_rows = write_split_shards(self.out_root, 'train', 'positive', safe_group, train_samples, self.shard_size)
    valid_rows = write_split_shards(self.out_root, 'valid', 'positive', safe_group, valid_samples, self.shard_size)
    return train_rows, valid_rows

  def __len__(self):
    return len(self.sta_date_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--shard_size', type=int, default=1024)
    args = parser.parse_args()
    event_list, _ = read_fpha(args.fpha, include_pick_metadata=True)
    sta_date_items = list(get_sta_date(event_list).items())
    train_rows, valid_rows = [], []
    dataset = Positive(sta_date_items, args.data_dir, args.out_root, args.shard_size)
    planned_by_item = [
        sum(pick_num_aug if samp_class == 'train' else 1
            for samp_class, _, _, _, pick_num_aug in samples)
        for _, samples in sta_date_items
    ]
    planned_attempts = sum(planned_by_item)
    processed_attempts = 0
    generated_samples = 0
    write_cut_progress(
        args.out_root, 'positive', 0, len(dataset), 0,
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
        if i % 10 == 0:
            print('%s/%s sta-date pairs done/total' % (i + 1, len(dataset)), flush=True)
            write_cut_progress(
                args.out_root, 'positive', i + 1, len(dataset),
                processed_attempts, planned_attempts, generated_samples,
            )
    train_rows.sort(key=lambda item: item[0])
    valid_rows.sort(key=lambda item: item[0])
    save_shard_index(os.path.join(args.out_root, 'train_pos.npy'), train_rows)
    save_shard_index(os.path.join(args.out_root, 'valid_pos.npy'), valid_rows)
    write_cut_progress(
        args.out_root, 'positive', len(dataset), len(dataset),
        processed_attempts, planned_attempts, generated_samples,
        finished=True,
    )
