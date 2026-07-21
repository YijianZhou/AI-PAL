""" Cut negative samples
    1. use num of dropped PAL picks to determine the num of neg to cut on each sta-date
    2. rand slice win on that sta-date
"""
import os, glob, shutil
import argparse
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from signal_lib import preprocess, sac_ch_time
from reader import dtime2str
import config
import warnings
warnings.filterwarnings("ignore")

# cut params
cfg = config.Config()
to_prep = cfg.to_prep
samp_rate = cfg.samp_rate
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
win_len = cfg.win_len
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
        sta_date = '%s_%s'%(net_sta, tp.date)
        if sta_date not in pick_dict: pick_dict[sta_date] = [[tp,ts]]
        else: pick_dict[sta_date].append([tp, ts])
    return pick_dict

def cut_event_window(stream_paths, t0, t1, out_paths):
    st  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
    if 0 in st.max() or len(st)!=3: return False
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st)!=3: return False
    # check FN
    amax_sec = [np.argmax(abs(tr.data))/samp_rate for tr in st]
    if min(amax_sec)>win_len/2: return False 
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    st = sac_ch_time(st)
    for ii, tr in enumerate(st): 
        tr.data[np.isnan(tr.data)] = 0
        tr.data[np.isinf(tr.data)] = 0
        tr.write(out_paths[ii], format='sac')
    return True

class Negative(Dataset):
  """ Dataset for cutting negative samples
  """
  def __init__(self, pick_num_items, pick_dict, cut_neg_ratio, data_dir, out_root):
    self.pick_num_items = pick_num_items
    self.pick_dict =  pick_dict
    self.cut_neg_ratio = cut_neg_ratio
    self.data_dir= data_dir
    self.out_root = out_root

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get one sta-date 
    sta_date, [num_unassoc, num_assoc] = self.pick_num_items[index]
    net_sta, date = sta_date.split('_')
    date = UTCDateTime(date)
    data_dict = get_data_dict(date, self.data_dir)
    if net_sta not in data_dict or num_unassoc==0: return train_paths_i, valid_paths_i
    stream_paths = data_dict[net_sta]
    out_train = os.path.join(self.out_root, 'train', 'negative', sta_date)
    out_valid = os.path.join(self.out_root, 'valid', 'negative', sta_date)
    if not os.path.exists(out_train): os.makedirs(out_train)
    if not os.path.exists(out_valid): os.makedirs(out_valid)
    # set num_cut
    assoc_ratio = num_assoc / (num_unassoc + num_assoc)
    if assoc_ratio>=max_assoc_ratio: return train_paths_i, valid_paths_i
    num_cut = int(num_unassoc * self.cut_neg_ratio * (max_assoc_ratio - assoc_ratio) / max_assoc_ratio)
    # get picks
    dtype = [('tp','O'),('ts','O')]
    picks = self.pick_dict[sta_date] if sta_date in self.pick_dict else []
    picks = np.array([(tp,ts) for tp,ts in picks], dtype=dtype)
    # cut events
    for _ in range(num_cut):
        # divide into train / valid
        rand = np.random.rand(1)[0]
        if rand<train_ratio: samp_class = 'train'
        elif rand<train_ratio+valid_ratio: samp_class = 'valid'
        else: continue
        # set out path
        out_dir = os.path.join(self.out_root, samp_class, 'negative', sta_date)
        start_time = date + win_len/2 + np.random.rand(1)[0]*(86400-win_len*1.5)
        end_time = start_time + win_len
        samp_name = 'neg_%s_%s'%(net_sta,dtime2str(start_time))
        out_paths = [os.path.join(out_dir,'0.%s.%s.sac'%(samp_name,ii+1)) for ii in range(3)]
        # check if tp-ts exists in selected win
        is_tp = (picks['tp']>start_time) * (picks['tp']<end_time)
        is_ts = (picks['ts']>start_time) * (picks['ts']<end_time)
        if sum(is_tp*is_ts)>0: continue
        # cut event window
        is_cut = cut_event_window(stream_paths, start_time, end_time, out_paths)
        if not is_cut: continue
        # record out_paths
        if samp_class=='train': train_paths_i.append(out_paths)
        if samp_class=='valid': valid_paths_i.append(out_paths)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.pick_num_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--fpick', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    # i/o paths
    train_root = os.path.join(args.out_root,'train')
    valid_root = os.path.join(args.out_root,'valid')
    fout_train_paths = os.path.join(args.out_root,'train_neg.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_neg.npy')
    # read fpha & fpick
    event_list, num_pos = read_fpha(args.fpha)
    pick_dict = get_pick_dict(event_list)
    pick_num_dict, num_picks = read_fpick(args.fpick, args.fpha)
    pick_num_items = list(pick_num_dict.items())
    cut_neg_ratio = (num_aug * num_pos) / (num_picks - num_pos)
    # for dates
    train_paths, valid_paths = [], []
    dataset = Negative(pick_num_items, pick_dict, cut_neg_ratio, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader):
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%100==0: print('%s/%s sta-date pairs done/total'%(i,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)