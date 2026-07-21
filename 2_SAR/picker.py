import os, glob
import time
import torch
import torch.nn.functional as F
import numpy as np
from obspy import UTCDateTime
from models import SAR
import config
import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()
# model config
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = cfg.win_len
win_len_npts = int(win_len * samp_rate)
win_stride = cfg.win_stride
win_stride_npts = int(win_stride * samp_rate)
step_len = cfg.rnn_step_len
step_len_npts = int(step_len * samp_rate)
step_stride = cfg.rnn_step_stride
step_stride_npts = int(step_stride * samp_rate)
num_steps = cfg.rnn_num_steps
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
# picker config
trig_thres = cfg.trig_thres
batch_size = cfg.picker_batch_size
tp_dev = cfg.tp_dev
ts_dev = cfg.ts_dev
amp_win = cfg.amp_win
amp_win_npts = int(sum(amp_win)*samp_rate)
rm_glitch = cfg.rm_glitch
amp_ratio_thres = cfg.amp_ratio_thres
win_peak = cfg.win_peak
win_peak_npts = int(win_peak * samp_rate)

class SAR_Picker(object):
  """ SAR picker for raw stream data
  """
  def __init__(self, ckpt_dir, ckpt_idx=-1, gpu_idx=0):
    if int(ckpt_idx)==-1:
        ckpt_idx = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, '*.ckpt'))])
    ckpt_idx = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt'%ckpt_idx)))[0] 
    print('SAR checkpoint: %s'%ckpt_idx)
    # load model
    self.device = torch.device("cuda:%s"%gpu_idx)
    self.model = SAR()
    self.model.load_state_dict(torch.load(ckpt_idx, map_location=self.device))
    self.model.to(self.device)
    self.model.eval()

  def pick(self, stream, fout=None):
    # 1. preprocess stream data 
    print('1. preprocess stream data')
    t = time.time()
    stream, st_raw = self.preprocess(stream)
    if len(stream)!=num_chn: return 
    start_time, end_time = stream[0].stats.starttime+win_stride, stream[0].stats.endtime
    if end_time < start_time + win_len: return
    stream, st_raw = stream.slice(start_time, end_time), st_raw.slice(start_time, end_time)
    net_sta = '%s.%s'%(stream[0].stats.network, stream[0].stats.station)
    num_win = int((end_time - start_time - win_len) / win_stride) + 1
    st_len_npts = min([len(trace) for trace in stream])
    st_data = np.array([trace.data[0:st_len_npts] for trace in stream], dtype=np.float32)
    st_data_cuda = torch.from_numpy(st_data).cuda(device=self.device)
    # find miss chn
    st_raw_npts = min([len(tr) for tr in st_raw])
    st_raw_data = np.array([tr.data[0:st_raw_npts] for tr in st_raw])
    raw_stride = int(st_raw[0].stats.sampling_rate * win_stride)
    raw_win_npts = int(st_raw[0].stats.sampling_rate * win_len)
    miss_chn = np.array([np.sum(st_raw_data[:, i*raw_stride : i*raw_stride+raw_win_npts]==0, axis=1)>win_len_npts/4 for i in range(num_win)])
    # 2. run SAR picker
    picks_raw = self.run_sar(st_data_cuda, start_time, num_win, miss_chn)
    num_picks = len(picks_raw)
    # 3.1 merge sliding-win picks
    print('3. merge sliding-win picks')
    to_drop = []
    for ii in range(num_picks):
        is_nbr = (abs(picks_raw['tp'] - picks_raw['tp'][ii]) < tp_dev) \
               * (abs(picks_raw['ts'] - picks_raw['ts'][ii]) < ts_dev)
        if sum(is_nbr)==1: continue
        prob_max = np.amax(picks_raw[is_nbr]['p_prob'] + picks_raw[is_nbr]['s_prob'])
        if picks_raw[ii]['p_prob'] + picks_raw[ii]['s_prob'] != prob_max: to_drop.append(ii)
    picks_raw = np.delete(picks_raw, to_drop)
    print('  %s picks dropped'%len(to_drop))
    # 3.2 get s_amp & glitch removal
    print('  get s_amp & glitch removal')
    picks = []
    for [tp, ts, p_prob, s_prob] in picks_raw:
        st = stream.slice(tp-amp_win[0], ts+amp_win[1]).copy()
        amp_data = np.array([tr.data[0:amp_win_npts] for tr in st])
        s_amp = self.get_s_amp(amp_data)
        if rm_glitch and self.remove_glitch(stream, tp, ts): continue
        picks.append([net_sta, tp, ts, s_amp, p_prob, s_prob])
        if fout:
            fout.write('{},{},{},{},{:.2f},{:.2f}\n'.format(net_sta, tp, ts, s_amp, p_prob, s_prob))
    print('total run time {:.2f}s'.format(time.time()-t))
    if not fout: return picks

  def run_sar(self, st_data_cuda, start_time, num_win, miss_chn):
    print('2. run SAR for phase picking')
    t = time.time()
    num_batch = int(np.ceil(num_win / batch_size))
    picks_raw = []
    dtype = [('tp','O'),('ts','O'),('p_prob','O'),('s_prob','O')]
    st_win_cuda = st_data_cuda.unfold(1, win_len_npts, win_stride_npts).permute(1, 0, 2)
    for batch_idx in range(num_batch):
        # get win_data
        n_win = batch_size if batch_idx<num_batch-1 else num_win%batch_size
        if n_win==0: n_win = batch_size
        win_idx_list = [nn + batch_idx*batch_size for nn in range(n_win)]
        data_seq = self.st2seq(st_win_cuda, win_idx_list, miss_chn)
        with torch.inference_mode():
            pred_logits = self.model(data_seq)
            pred_probs = F.softmax(pred_logits, dim=-1).detach().cpu().numpy()
        # decode to sec
        for nn, pred_prob in enumerate(pred_probs):
            win_idx = nn + batch_idx*batch_size
            t0 = start_time + win_idx * win_stride
            if sum(miss_chn[win_idx])==3: continue
            pred_prob_p, pred_prob_s = pred_prob[:,1], pred_prob[:,2]
            pred_prob_p[np.isnan(pred_prob_p)] = 0
            pred_prob_s[np.isnan(pred_prob_s)] = 0
            if min(np.amax(pred_prob_p), np.amax(pred_prob_s)) < trig_thres: continue
            p_idxs = np.where(pred_prob_p>=trig_thres)[0]
            s_idxs = np.where(pred_prob_s>=trig_thres)[0]
            p_dets = np.split(p_idxs, np.where(np.diff(p_idxs)!=1)[0] + 1)
            s_dets = np.split(s_idxs, np.where(np.diff(s_idxs)!=1)[0] + 1)
            p_probs = [np.amax(pred_prob_p[p_det]) for p_det in p_dets]
            s_probs = [np.amax(pred_prob_s[s_det]) for s_det in s_dets]
            p_idxs = [np.median(x) for x in p_dets]
            s_idxs = [np.median(x) for x in s_dets]
            for ii, p_idx in enumerate(p_idxs):
                tp = t0 + step_len/2 + step_stride*p_idx
                p_prob = p_probs[ii]
                for jj, s_idx in enumerate(s_idxs):
                    if s_idx<=p_idx: continue
                    ts = t0 + step_len/2 + step_stride*s_idx
                    s_prob = s_probs[jj]
                    picks_raw.append((tp, ts, p_prob, s_prob))
    print('  {} raw P&S picks | SAR run time {:.2f}s'.format(len(picks_raw), time.time()-t))
    return np.array(picks_raw, dtype=dtype)

  def st2seq(self, st_win_cuda, win_idx_list, miss_chn):
    win_idx_tensor = torch.as_tensor(win_idx_list, dtype=torch.long, device=self.device)
    win_data = st_win_cuda.index_select(0, win_idx_tensor).contiguous()
    win_data = self.preprocess_cuda_batch(win_data, miss_chn[win_idx_list])
    data_seq = win_data.unfold(2, step_len_npts, step_stride_npts)
    data_seq = data_seq.permute(0, 2, 1, 3).reshape(win_data.size(0), -1, num_chn*step_len_npts)
    if data_seq.size(1) < num_steps:
        pad = data_seq.new_zeros(data_seq.size(0), num_steps-data_seq.size(1), data_seq.size(2))
        data_seq = torch.cat((data_seq, pad), dim=1)
    elif data_seq.size(1) > num_steps:
        data_seq = data_seq[:, 0:num_steps, :]
    return data_seq.contiguous()

  def preprocess(self, st, max_gap=5.):
    # align time
    if len(st)!=num_chn: return [], []
    start_time = max([tr.stats.starttime for tr in st])
    end_time = min([tr.stats.endtime for tr in st])
    if end_time < start_time + win_len: return [], []
    st = st.slice(start_time, end_time, nearest_sample=True)
    if len(st)!=num_chn: return [], []
    # remove nan & inf
    for ii in range(3):
        st[ii].data[np.isnan(st[ii].data)] = 0
        st[ii].data[np.isinf(st[ii].data)] = 0
    if max(st.max())==0: return [], []
    st_raw = st.copy()
    # fill data gap
    max_gap_npts = int(max_gap*samp_rate)
    for tr in st:
        npts = len(tr.data)
        data_diff = np.diff(tr.data)
        gap_idx = np.where(data_diff==0)[0]
        gap_list = np.split(gap_idx, np.where(np.diff(gap_idx)!=1)[0] + 1)
        gap_list = [gap for gap in gap_list if len(gap)>=10]
        num_gap = len(gap_list)
        for ii,gap in enumerate(gap_list):
            idx0, idx1 = max(0, gap[0]-1), min(npts-1, gap[-1]+1)
            if ii<num_gap-1: idx2 = min(idx1+(idx1-idx0), idx1+max_gap_npts, gap_list[ii+1][0])
            else: idx2 = min(idx1+(idx1-idx0), idx1+max_gap_npts, npts-1)
            if idx1==idx2: continue
            if idx2==idx1+(idx1-idx0): tr.data[idx0:idx1] = tr.data[idx1:idx2]
            else:
                num_tile = int(np.ceil((idx1-idx0)/(idx2-idx1)))
                tr.data[idx0:idx1] = np.tile(tr.data[idx1:idx2], num_tile)[0:idx1-idx0]
    # resample 
    st = st.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=5.)
    org_rate = st[0].stats.sampling_rate
    if org_rate!=samp_rate: st.resample(samp_rate)
    # filter
    freq_min, freq_max = freq_band
    if freq_min and freq_max:
        return st.filter('bandpass', freqmin=freq_min, freqmax=freq_max), st_raw
    elif not freq_max and freq_min: 
        return st.filter('highpass', freq=freq_min), st_raw
    elif not freq_min and freq_max: 
        return st.filter('lowpass', freq=freq_max), st_raw
    else:
        print('filter type not supported!'); return [], []

  # preprocess cuda data (in-place)
  def preprocess_cuda(self, data, is_miss):
    # fix missed channel
    if 0<sum(is_miss)<3: data[is_miss] = data[~is_miss][-1]
    # rmean & norm
    data -= torch.mean(data, axis=1).view(num_chn,1)
    if global_max_norm: data /= torch.max(abs(data)).clamp_min(1e-12)
    else: data /= torch.max(abs(data), axis=1).values.clamp_min(1e-12).view(num_chn,1)
    return data

  def preprocess_cuda_batch(self, data, miss_chn_batch):
    # data: (num_win, num_chn, win_len_npts)
    miss_chn_batch = np.asarray(miss_chn_batch, dtype=bool)
    miss_count = np.sum(miss_chn_batch, axis=1)
    repair_rows = np.where((miss_count > 0) & (miss_count < num_chn))[0]
    for row in repair_rows:
        miss = torch.as_tensor(miss_chn_batch[row], dtype=torch.bool, device=self.device)
        data[row, miss] = data[row, ~miss][-1]
    data -= torch.mean(data, dim=2, keepdim=True)
    if global_max_norm:
        scale = torch.amax(torch.abs(data), dim=(1, 2), keepdim=True)
    else:
        scale = torch.amax(torch.abs(data), dim=2, keepdim=True)
    data /= scale.clamp_min(1e-12)
    return data

  # get S amplitide
  def get_s_amp(self, velo):
    velo -= np.reshape(np.mean(velo, axis=1), [velo.shape[0],1])
    disp = np.cumsum(velo, axis=1)
    disp /= samp_rate
    return np.amax(np.sum(disp**2, axis=0))**0.5

  # glitch removal based on PAL algorithm 
  def remove_glitch(self, stream, tp, ts):
    # 1. Peak amp ratio of P&S
    p_amp_ratio = self.calc_peak_amp_ratio(stream.slice(tp, tp+win_peak*3))
    if np.amin(p_amp_ratio) > amp_ratio_thres[0]: return True
    s_amp_ratio = self.calc_peak_amp_ratio(stream.slice(ts, ts+win_peak*3))
    if np.amin(s_amp_ratio) > amp_ratio_thres[0]: return True
    # 2. amp ratio of P/P_tail & S
    A1 = np.array([np.amax(tr.data)-np.amin(tr.data) for tr in stream.slice(tp, tp+(ts-tp)/2)])
    A2 = np.array([np.amax(tr.data)-np.amin(tr.data) for tr in stream.slice(tp+(ts-tp)/2, ts)])
    A3 = np.array([np.amax(tr.data)-np.amin(tr.data) for tr in stream.slice(ts, ts+(ts-tp)/2)])
    A12 = min([A1[ii]/A2[ii] for ii in range(3)])
    A13 = min([A1[ii]/A3[ii] for ii in range(3)])
    if A12<amp_ratio_thres[1] and A13<amp_ratio_thres[2]: return False
    else: return True

  def calc_peak_amp_ratio(self, st):
    # find peak idx
    peak_data = np.array([abs(tr.data[0:win_peak_npts]) for tr in st])
    chn_idx = np.unravel_index(np.argmax(peak_data), peak_data.shape)[0]
    idx0 = np.argmax(abs(st[chn_idx].data[0:win_peak_npts]))
    idx1 = idx0 + self.find_first_peak(st[chn_idx].data[idx0:])
    idx0 -= self.find_second_peak(st[chn_idx].data[0:idx0][::-1])
    idx1 += self.find_second_peak(st[chn_idx].data[idx1:])+1
    idx0 = max(0,idx0)
    # calc peak amp ratio 
    amp_ratio = []
    for tr in st:
        amp_peak = np.amax(tr.data[idx0:idx1]) - np.amin(tr.data[idx0:idx1])
        amp_tail = np.amax(tr.data[idx1:2*idx1-idx0]) - np.amin(tr.data[idx1:2*idx1-idx0])
        amp_ratio.append(amp_peak/amp_tail)
    return amp_ratio

  def find_first_peak(self, data):
    npts = len(data)
    if npts<2: return 0
    delta_d = data[1:npts] - data[0:npts-1]
    if min(delta_d)>=0 or max(delta_d)<=0: return 0
    neg_idx = np.where(delta_d<0)[0]
    pos_idx = np.where(delta_d>=0)[0]
    return max(neg_idx[0], pos_idx[0])

  def find_second_peak(self, data):
    npts = len(data)
    if npts<2: return 0
    delta_d = data[1:npts] - data[0:npts-1]
    if min(delta_d)>=0 or max(delta_d)<=0: return 0
    neg_idx = np.where(delta_d<0)[0]
    pos_idx = np.where(delta_d>=0)[0]
    if len(neg_idx)==0 or len(pos_idx)==0: return 0
    first_peak = max(neg_idx[0], pos_idx[0])
    neg_peak = neg_idx[neg_idx>first_peak]
    pos_peak = pos_idx[pos_idx>first_peak]
    if len(neg_peak)==0 or len(pos_peak)==0: return first_peak
    return max(neg_peak[0], pos_peak[0])