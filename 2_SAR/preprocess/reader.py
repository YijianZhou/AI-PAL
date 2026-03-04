""" File reader
"""
import os
import glob
import numpy as np
from obspy import read, UTCDateTime

# read phase file
def read_fpha(fpha):
    f=open(fpha); lines=f.readlines(); f.close()
    event_list, num_pos = [], 0
    for line in lines:
        codes = line.split(',')
        if len(codes[0])>10:
            ot = UTCDateTime(codes[0])
            lat, lon, dep, mag = [float(code) for code in codes[1:5]]
            event_loc = [ot, lat, lon, dep, mag]
            event_list.append([event_loc, {}])
        else:
            net_sta = codes[0]
            tp, ts = [UTCDateTime(code) for code in codes[1:3]]
            event_list[-1][-1][net_sta] = [tp, ts]
            num_pos += 1
    return event_list, num_pos

# read pick file to get number of dropped picks
def read_fpick(fpick, fpha=None):
    # get associated picks
    print('read fpha to find associated picks')
    pick_assoc_dict = {}
    if fpha:
        f=open(fpha); lines=f.readlines(); f.close()
        for line in lines:
            codes = line.split(',')
            if len(codes[0])>10: continue
            net_sta, tp_str = codes[0:2]
            date = str(UTCDateTime(tp_str).date)
            if date not in pick_assoc_dict: pick_assoc_dict[date] = [tp_str]
            else: pick_assoc_dict[date].append(tp_str)
    # group picks by sta-date
    print('read fpick and remove associated picks')
    pick_num_dict, num_picks = {}, 0
    f=open(fpick); lines=f.readlines(); f.close()
    for ii,line in enumerate(lines):
        if ii%1e5==0: print('%s/%s lines done'%(ii,len(lines)))
        codes = line.split(',')
        net_sta, tp_str = codes[0], codes[2]  # PAL format
        date = str(UTCDateTime(tp_str).date)
        sta_date = '%s_%s'%(net_sta, date)
        if sta_date not in pick_num_dict: pick_num_dict[sta_date] = [0,0] # unassoc, assoc
        if date in pick_assoc_dict and tp_str in pick_assoc_dict[date]: 
            pick_num_dict[sta_date][1] += 1
        else: pick_num_dict[sta_date][0] += 1
        num_picks += 1
    return pick_num_dict, num_picks

# read PAL station file
def get_sta_dict(fsta):
    sta_dict = {}
    f=open(fsta); lines=f.readlines(); f.close()
    for line in lines:
        codes = line.split(',')
        net_sta = codes[0]
        lat, lon, ele = [float(code) for code in codes[1:4]]
        # format 1: same gain for 3-chn & time invariant
        if len(codes[4:])==1: gain = float(codes[4])
        # format 2: different gain for 3-chn & time invariant
        elif len(codes[4:])==3: gain = [float(code) for code in codes[4:]]
        # format 3: different gain for 3-chn & time variant
        elif len(codes[4:])==5:
            gain = [float(code) for code in codes[4:7]]
            gain += [UTCDateTime(code) for code in codes[7:9]]
            gain = [gain]
        else: print('false sta_file format!'); continue
        if net_sta not in sta_dict:
           sta_dict[net_sta] = [lat,lon,ele,gain]
        else:
           sta_dict[net_sta][-1].append(gain[0]) # if format 3
    return sta_dict

# get data dict, given path structure
def get_data_dict(date, data_dir):
    # get data paths
    data_dict = {}
    date_dir = '{:0>4}{:0>2}{:0>2}'.format(date.year, date.month, date.day)
    st_paths = sorted(glob.glob(os.path.join(data_dir, date_dir, '*')))
    for st_path in st_paths:
        fname = os.path.split(st_path)[-1]
        net_sta = '.'.join(fname.split('.')[0:2])
        if net_sta in data_dict: data_dict[net_sta].append(st_path)
        else: data_dict[net_sta] = [st_path]
    # drop bad sta
    todel = [net_sta for net_sta in data_dict if len(data_dict[net_sta])!=3]
    for net_sta in todel: data_dict.pop(net_sta)
    return data_dict

# if your data contains single-channel stations
def get_1chn_data(date, data_dir):
    # get data paths
    data_dict = {}
    date_code = '{:0>4}{:0>2}{:0>2}'.format(date.year, date.month, date.day)
    st_paths = sorted(glob.glob(os.path.join(data_dir, date_code, '*')))
    for st_path in st_paths:
        fname = os.path.basename(st_path)
        net_sta = '.'.join(fname.split('.')[0:2])
        if net_sta in data_dict: data_dict[net_sta].append(st_path)
        else: data_dict[net_sta] = [st_path]
    # expand single channel station
    for net_sta in data_dict:
        if len(data_dict[net_sta])!=3: data_dict[net_sta] = [data_dict[net_sta][-1]]*3
    return data_dict

# read stream data
def read_data(st_paths, sta_dict):
    # read data
    print('reading stream: {}'.format(st_paths[0]))
    try:
        st  = read(st_paths[0])
        st += read(st_paths[1])
        st += read(st_paths[2])
    except:
        print('bad data!'); return []
    # change header
    net, sta = os.path.basename(st_paths[0]).split('.')[0:2]
    net_sta = '%s.%s'%(net,sta)
    gain = sta_dict[net_sta][3]
    start_time = max([tr.stats.starttime for tr in st])
    end_time = min([tr.stats.endtime for tr in st])
    st_time = start_time + (end_time-start_time)/2
    for ii in range(3): st[ii].stats.network, st[ii].stats.station = net, sta
    # if format 1: same gain for 3-chn & time invariant
    if type(gain)==float:
        for ii in range(3): st[ii].data = st[ii].data / gain
    # if format 2: different gain for 3-chn & time invariant
    elif type(gain[0])==float:
        for ii in range(3): st[ii].data = st[ii].data / gain[ii]
    elif type(gain[0])==list:
        for [ge,gn,gz,t0,t1] in gain:
            if t0<st_time<t1: break
        for ii in range(3): st[ii].data = st[ii].data / [ge,gn,gz][ii]
    return st

# UTCDateTime to string
def dtime2str(dtime):
    date = ''.join(str(dtime).split('T')[0].split('-'))
    time = ''.join(str(dtime).split('T')[1].split(':'))[0:9]
    return date + time

""" Custimized functions
"""
