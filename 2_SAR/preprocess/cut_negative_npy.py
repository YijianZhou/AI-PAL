"""Cut negative SAR training samples directly into raw-waveform NPY shards."""
import os
import argparse
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from signal_lib import preprocess, sac_ch_time
from reader import dtime2str
from dataset_npy import stream_to_sample, write_split_shards, save_shard_index
import config
import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()
to_prep = cfg.to_prep
samp_rate = cfg.samp_rate
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
win_len = cfg.win_len
win_npts = int(win_len * samp_rate)
read_fpha = cfg.read_fpha
read_fpick = cfg.read_fpick
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
max_assoc_ratio = cfg.max_assoc_ratio
num_aug = cfg.num_aug


def get_pick_dict(event_list):
    pick_dict = {}
    for i, [_, picks] in enumerate(event_list):
      for net_sta, [tp, ts] in picks.items():
        sta_date = '%s_%s' % (net_sta, tp.date)
        pick_dict.setdefault(sta_date, []).append([tp, ts])
    return pick_dict


def cut_event_window(stream_paths, t0, t1):
    st  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
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
    st = sac_ch_time(st)
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
    data_dict = get_data_dict(date, self.data_dir)
    if net_sta not in data_dict or num_unassoc == 0:
        return [], []
    stream_paths = data_dict[net_sta]
    assoc_ratio = num_assoc / (num_unassoc + num_assoc)
    if assoc_ratio >= max_assoc_ratio:
        return [], []
    num_cut = int(num_unassoc * self.cut_neg_ratio * (max_assoc_ratio - assoc_ratio) / max_assoc_ratio)
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
        is_tp = (picks['tp'] > start_time) * (picks['tp'] < end_time)
        is_ts = (picks['ts'] > start_time) * (picks['ts'] < end_time)
        if sum(is_tp*is_ts) > 0:
            continue
        st = cut_event_window(stream_paths, start_time, end_time)
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
    parser.add_argument('--fpick', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--shard_size', type=int, default=1024)
    args = parser.parse_args()
    event_list, num_pos = read_fpha(args.fpha)
    pick_dict = get_pick_dict(event_list)
    pick_num_dict, num_picks = read_fpick(args.fpick, args.fpha)
    pick_num_items = list(pick_num_dict.items())
    cut_neg_ratio = (num_aug * num_pos) / (num_picks - num_pos)
    train_rows, valid_rows = [], []
    dataset = Negative(pick_num_items, pick_dict, cut_neg_ratio, args.data_dir, args.out_root, args.shard_size)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i, [train_rows_i, valid_rows_i] in enumerate(dataloader):
        train_rows += train_rows_i
        valid_rows += valid_rows_i
        if i % 100 == 0:
            print('%s/%s sta-date pairs done/total' % (i, len(dataset)), flush=True)
    train_rows.sort(key=lambda item: item[0])
    valid_rows.sort(key=lambda item: item[0])
    save_shard_index(os.path.join(args.out_root, 'train_neg.npy'), train_rows)
    save_shard_index(os.path.join(args.out_root, 'valid_neg.npy'), valid_rows)