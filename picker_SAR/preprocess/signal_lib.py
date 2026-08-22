""" Signal processing library
"""
import numpy as np
from obspy import read, UTCDateTime

def preprocess(stream, samp_rate, freq_band, max_gap=5.):
    # time alignment
    start_time = max([trace.stats.starttime for trace in stream])
    end_time = min([trace.stats.endtime for trace in stream])
    if start_time>=end_time: print('bad data!'); return []
    st = stream.slice(start_time, end_time)
    # remove nan & inf
    for tr in st:
        tr.data[np.isnan(tr.data)] = 0
        tr.data[np.isinf(tr.data)] = 0
    # fill data gap
    max_gap_npts = int(max_gap*samp_rate)
    for tr in st:
        npts = len(tr.data)
        data_diff = np.diff(tr.data)
        gap_idx = np.where(data_diff==0)[0]
        gap_list = np.split(gap_idx, np.where(np.diff(gap_idx)!=1)[0] + 1)
        gap_list = [gap for gap in gap_list if len(gap)>=3]
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
        return st.filter('bandpass', freqmin=freq_min, freqmax=freq_max)
    elif not freq_max and freq_min:
        return st.filter('highpass', freq=freq_min)
    elif not freq_min and freq_max:
        return st.filter('lowpass', freq=freq_max)
    else:
        print('filter type not supported!'); return []

def sac_ch_time(st):
    for tr in st:
        t0 = tr.stats.starttime
        if not 'sac' in tr.stats: continue
        tr.stats.sac.nzyear = t0.year
        tr.stats.sac.nzjday = t0.julday
        tr.stats.sac.nzhour = t0.hour
        tr.stats.sac.nzmin = t0.minute
        tr.stats.sac.nzsec = t0.second
        tr.stats.sac.nzmsec = int(t0.microsecond / 1e3)
    return st

