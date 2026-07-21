""" Cut positive samples 
    1. for all sta-date pairs
    2. cut all events in that sta-date, use real data for noise aug 
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
samp_rate = cfg.samp_rate
win_len = cfg.win_len
step_len = cfg.rnn_step_len
rand_dt_max = win_len/2 - step_len  # rand before P
read_fpha = cfg.read_fpha
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
freq_band = cfg.freq_band
to_prep = cfg.to_prep
global_max_norm = cfg.global_max_norm
num_aug = cfg.num_aug
max_noise = cfg.max_noise

def get_sta_date(event_list):
    sta_date_dict = {}
    for [event_loc, picks] in event_list:
        # 1. get event info
        ot, lat, lon = event_loc[0:3]
        event_name = dtime2str(ot)
        train_dir = os.path.join(train_root, 'positive', event_name)
        valid_dir = os.path.join(valid_root, 'positive', event_name)
        if not os.path.exists(train_dir): os.makedirs(train_dir)
        if not os.path.exists(valid_dir): os.makedirs(valid_dir)
        for net_sta, [tp, ts] in picks.items():
            # 2. divide into train / valid
            rand = np.random.rand(1)[0]
            if rand<train_ratio: samp_class = 'train'
            elif rand<train_ratio+valid_ratio: samp_class = 'valid'
            else: continue
            sta_date = '%s_%s'%(net_sta, tp.date) 
            if sta_date not in sta_date_dict:
                sta_date_dict[sta_date] = [[samp_class, event_name, tp, ts]]
            else: sta_date_dict[sta_date].append([samp_class, event_name, tp, ts])
    return sta_date_dict

def add_noise(st, stream_paths, tp, ts, picks):
    # find noise win
    date = UTCDateTime(st[0].stats.starttime.date)
    t0 = date + win_len/2 + np.random.rand(1)[0] * (86400-win_len*1.5)
    t1 = t0 + win_len
    # check if tp-ts exists in selected win
    is_tp = (picks['tp']>t0) * (picks['tp']<t1)
    is_ts = (picks['ts']>t0) * (picks['ts']<t1)
    if sum(is_tp*is_ts)>0: return st
    # add noise from real data
    st_noise  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st_noise += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st_noise += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
    if len(st_noise)!=3: return st
    if to_prep: st_noise = preprocess(st_noise, samp_rate, freq_band)
    st_noise = st_noise.slice(t0, t1).normalize(global_max=global_max_norm)
    if len(st_noise)!=3: return st
    npts = min([len(tr) for tr in st+st_noise])
    noise_scale = max_noise * np.random.rand(1)[0]
    for ii in range(3):
        scale = noise_scale * np.amax(abs(st[ii].slice(tp, ts).data))
        st[ii].data[0:npts] += st_noise[ii].data[0:npts] * scale
    return st.detrend('demean').normalize(global_max=global_max_norm)

def cut_event_window(stream_paths, t0, t1):
    st  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
    if 0 in st.max() or len(st)!=3: return None
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st)!=3: return None
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    return st
    
class Positive(Dataset):
  """ Dataset for cutting positive samples
  """
  def __init__(self, sta_date_items, data_dir, out_root):
    self.sta_date_items = sta_date_items
    self.data_dir= data_dir
    self.out_root = out_root

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get one sta-date
    sta_date, samples = self.sta_date_items[index]
    net_sta, date = sta_date.split('_')
    data_dict = get_data_dict(UTCDateTime(date), self.data_dir)
    if net_sta not in data_dict: return train_paths_i, valid_paths_i
    stream_paths = data_dict[net_sta]
    # get picks
    dtype = [('tp','O'),('ts','O')]
    picks = np.array([(tp,ts) for _,_,tp,ts in samples], dtype=dtype)
    # cut event win
    for [samp_class, event_name, tp, ts] in samples:
        if tp>ts: continue
        out_dir = os.path.join(self.out_root, samp_class, 'positive', event_name)
        samp_name = 'pos_%s_%s'%(net_sta, event_name[:-3])
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            # rand time shift & prep
            rand_dt = min(rand_dt_max, win_len-step_len-(ts-tp))
            start_time = tp - step_len - np.random.rand(1)[0] * rand_dt
            end_time = start_time + win_len
            sac_t0, sac_t1 = tp-start_time, ts-start_time
            st = cut_event_window(stream_paths, start_time, end_time)
            if not st: continue
            if aug_idx>0 and max_noise>0: st = add_noise(st, stream_paths, tp, ts, picks)
            # write stream
            st = sac_ch_time(st)
            out_paths = [os.path.join(out_dir,'%s.%s.%s.sac'%(aug_idx,samp_name,ii+1)) for ii in range(3)]
            for ii,tr in enumerate(st):
                tr.write(out_paths[ii], format='sac')
                tr = read(out_paths[ii])[0]
                tr.stats.sac.t0, tr.stats.sac.t1 = sac_t0, sac_t1
                tr.data[np.isnan(tr.data)] = 0
                tr.data[np.isinf(tr.data)] = 0
                tr.write(out_paths[ii], format='sac')
            # record out_paths
            if samp_class=='train': train_paths_i.append(out_paths)
            if samp_class=='valid': valid_paths_i.append(out_paths)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.sta_date_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    # i/o paths
    train_root = os.path.join(args.out_root,'train')
    valid_root = os.path.join(args.out_root,'valid')
    fout_train_paths = os.path.join(args.out_root,'train_pos.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_pos.npy')
    # read fpha
    event_list, _ = read_fpha(args.fpha)
    sta_date_dict = get_sta_date(event_list)
    sta_date_items = list(sta_date_dict.items())
    # for sta-date pairs
    train_paths, valid_paths = [], []
    dataset = Positive(sta_date_items, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader):
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%10==0: print('%s/%s sta-date pairs done/total'%(i,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)
