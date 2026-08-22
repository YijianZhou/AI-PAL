import os, glob
import time
import torch
import torch.nn.functional as F
import numpy as np
from obspy import UTCDateTime
from pick_ensemble import cluster_pair_votes, format_pick_row
try:
    from .models import UNet
    from . import config
except ImportError:
    from models import UNet
    import config
from data_pipeline import preprocess_picker_stream
from picker_stream import PreparedPickerStream
from waveform_qc import calc_peak_amp_ratio as calc_qc_peak_amp_ratio
from waveform_qc import remove_glitch as remove_waveform_glitch
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
#step_len = cfg.rnn_step_len
#step_len_npts = int(step_len * samp_rate)
#step_stride = cfg.rnn_step_stride
#step_stride_npts = int(step_stride * samp_rate)
#num_steps = cfg.rnn_num_steps
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
# picker config
trig_thres = cfg.trig_thres
batch_size = cfg.picker_batch_size
tp_dev = cfg.tp_dev
ts_dev = cfg.ts_dev
picker_min_cluster_size = cfg.picker_min_cluster_size
taper_max_length_sec = cfg.taper_max_length_sec
amp_win = cfg.amp_win
amp_win_npts = int(sum(amp_win)*samp_rate)
rm_glitch = cfg.rm_glitch
amp_ratio_thres = cfg.amp_ratio_thres
win_peak = cfg.win_peak
win_peak_npts = int(win_peak * samp_rate)

class RUN_Picker(object):
  """ResUNet picker for raw stream data
  """
  def __init__(self, ckpt_dir, ckpt_idx=-1, gpu_idx=0):
    if os.path.isfile(ckpt_dir):
        ckpt_path = ckpt_dir
    else:
        if int(ckpt_idx)==-1:
            ckpt_idx = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, '*.ckpt'))])
        ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt'%ckpt_idx)))[0]
    print('RUN checkpoint: %s'%ckpt_path)
    # load model
    gpu_idx = int(gpu_idx)
    if gpu_idx < 0:
        self.device = torch.device("cpu")
    elif torch.cuda.is_available():
        self.device = torch.device("cuda:{}".format(gpu_idx))
    else:
        raise RuntimeError(
            "GPU {} requested but CUDA is unavailable; set gpu_idx=-1 for CPU"
            .format(gpu_idx)
        )
    print("inference device: {}".format(self.device))
    self.amp_enabled = self.device.type == 'cuda' and bool(getattr(cfg, 'amp', True))
    self.amp_dtype = (
        torch.bfloat16 if self.amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    print(
        'RUN inference AMP: {}'.format(
            'disabled' if not self.amp_enabled else str(self.amp_dtype).replace('torch.', '')
        )
    )
    self.model = UNet()
    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
    self.model.to(self.device)
    self.model.eval()

  def pick(self, stream, fout=None, pick_start_time=None, pick_end_time=None,
           prepared=None, defer_waveform_qc=False):
    # 1. preprocess stream data 
    print('1. {} stream data'.format(
        'standalone preprocess' if prepared is None else 'use shared preprocessed'
    ))
    t = time.time()
    if prepared is None:
        stream, st_raw = self.preprocess(stream)
        if len(stream) != num_chn:
            return
        try:
            prepared = PreparedPickerStream(
                stream, st_raw, samp_rate, win_len, win_stride,
                taper_max_length_sec, num_channels=num_chn,
            )
        except ValueError:
            return
    else:
        prepared.validate_layout(samp_rate, win_len, win_stride, num_chn)
    stream = prepared.stream
    start_time = prepared.start_time
    end_time = prepared.end_time
    net_sta = prepared.net_sta
    num_win = prepared.num_windows
    st_data_cuda = prepared.tensor_for(self.device)
    miss_chn = prepared.missing_channels
    # 2. run ResUNet picker
    picks_raw = self.run_run(st_data_cuda, start_time, num_win, miss_chn)
    # 3.1 ensemble sliding-window P/S-pair picks
    print('3. cluster sliding-window picks')
    picks_raw = cluster_pair_votes(
        picks_raw, tp_dev, ts_dev,
        min_support=picker_min_cluster_size, source_field='win_idx',
    )
    print('  {} accepted clusters (minimum {} windows)'.format(
        len(picks_raw), picker_min_cluster_size
    ))
    # 3.2 waveform QC is deferred when association follows in memory.
    print('  {}'.format(
        'defer waveform QC until association'
        if defer_waveform_qc else 'get s_amp & glitch removal'
    ))
    picks = []
    for consensus in picks_raw:
        tp, ts = UTCDateTime(consensus['tp']), UTCDateTime(consensus['ts'])
        if pick_start_time is not None and tp < UTCDateTime(pick_start_time): continue
        if pick_end_time is not None and tp >= UTCDateTime(pick_end_time): continue
        p_prob, s_prob = consensus['p_prob'], consensus['s_prob']
        s_amp = -1.0
        if not defer_waveform_qc:
          st = stream.slice(tp-amp_win[0], ts+amp_win[1]).copy()
          amp_data = np.array([tr.data[0:amp_win_npts] for tr in st])
          s_amp = self.get_s_amp(amp_data)
          if rm_glitch and self.remove_glitch(stream, tp, ts): continue
        output = dict(consensus)
        output.update(
            net_sta=net_sta,
            tp=tp,
            ts=ts,
            s_amp=s_amp,
            sources=['RUN'],
            picker_cluster_sizes={'RUN': int(consensus['num_support'])},
        )
        picks.append(output)
        if fout:
            fout.write(format_pick_row(output))
    print('total run time {:.2f}s'.format(time.time()-t))
    return picks

  def run_run(self, st_data_cuda, start_time, num_win, miss_chn):
    print('2. run ResUNet for phase picking')
    t = time.time()
    num_batch = int(np.ceil(num_win / batch_size))
    picks_raw = []
    dtype = [('tp','O'),('ts','O'),('p_prob','O'),('s_prob','O'),('win_idx','i4')]
    st_win_cuda = st_data_cuda.unfold(1, win_len_npts, win_stride_npts).permute(1, 0, 2)
    for batch_idx in range(num_batch):
        n_win = batch_size if batch_idx < num_batch-1 else num_win % batch_size
        if n_win == 0: n_win = batch_size
        win_idx_list = [nn + batch_idx*batch_size for nn in range(n_win)]
        data_batch = self.st2win(st_win_cuda, win_idx_list, miss_chn)
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                pred_logits = self.model(data_batch)
            pred_probs = F.softmax(pred_logits.float(), dim=1).cpu().numpy()
        # decode to sec
        for nn, pred_prob in enumerate(pred_probs):
            win_idx = nn + batch_idx*batch_size
            t0 = start_time + win_idx * win_stride
            if sum(miss_chn[win_idx])==3: continue
            pred_prob_p, pred_prob_s = pred_prob[1], pred_prob[2]
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
                tp = t0 + p_idx/samp_rate
                p_prob = p_probs[ii]
                for jj, s_idx in enumerate(s_idxs):
                    if s_idx<=p_idx: continue
                    ts = t0 + s_idx/samp_rate
                    s_prob = s_probs[jj]
                    picks_raw.append((tp, ts, p_prob, s_prob, win_idx))
    print('  {} raw P&S picks | ResUNet run time {:.2f}s'.format(len(picks_raw), time.time()-t))
    return np.array(picks_raw, dtype=dtype)

  def st2win(self, st_win_cuda, win_idx_list, miss_chn):
    win_idx_tensor = torch.as_tensor(win_idx_list, dtype=torch.long, device=self.device)
    win_data = st_win_cuda.index_select(0, win_idx_tensor).contiguous()
    return self.preprocess_cuda_batch(win_data, miss_chn[win_idx_list])

  def preprocess(self, st, max_gap=5.):
    """Standalone preprocessing; ensemble runs pass a prepared waveform."""
    return preprocess_picker_stream(
      st,
      num_channels=num_chn,
      sampling_rate=samp_rate,
      min_length_sec=win_len,
      frequency_band=freq_band,
      taper_max_length_sec=taper_max_length_sec,
      max_gap_sec=max_gap,
    )
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
    return remove_waveform_glitch(
      stream, tp, ts, win_peak, win_peak_npts, amp_ratio_thres,
      self.find_first_peak, self.find_second_peak,
    )

  def calc_peak_amp_ratio(self, st):
    return calc_qc_peak_amp_ratio(
      st, win_peak_npts, self.find_first_peak, self.find_second_peak,
    )

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


# Backward-compatible alias for copied PhaseNet runner names.
PHN_Picker = RUN_Picker
