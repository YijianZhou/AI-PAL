""" Make phase file for ph2dt_cc (fpha_temp)
  fpha_name --> event_name --> event data dir
  fpha_ot --> ot
  fpha_loc --> lat, lon, dep, mag
"""
import numpy as np
from obspy import UTCDateTime
from dataset_cc import dtime2str
import config

# i/o paths
cfg = config.Config()
fpha_name = cfg.fpha_name
fpha_ot = cfg.fpha_ot
fpha_loc = cfg.fpha_loc
fout = open('input/phase.temp','w')
ot_min, ot_max = [UTCDateTime(date) for date in cfg.ot_range.split('-')]
lat_min, lat_max = cfg.lat_range
lon_min, lon_max = cfg.lon_range
xy_pad = cfg.xy_pad
lon_min -= xy_pad[0]
lon_max += xy_pad[0]
lat_min -= xy_pad[1]
lat_max += xy_pad[1]

# 1. get evid & event_name
event_dict = {}
f=open(fpha_name); lines=f.readlines(); f.close()
evid = 0
for line in lines:
    codes = line.split(',')
    if len(codes[0])<10: continue
    event_name = dtime2str(UTCDateTime(codes[0]))
    event_dict[str(evid)] = event_name
    evid += 1

# 2. get ot
ot_dict = {}
f=open(fpha_ot); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes[0])<10: continue
    evid = codes[-1][:-1]
    ot_dict[evid] = UTCDateTime(codes[0])

# 3. get loc
f=open(fpha_loc); lines=f.readlines(); f.close()
for line in lines:
    codes = line.split(',')
    if len(codes[0])>=10: 
        evid = codes[-1][:-1]
        event_name = event_dict[evid]
        id_name = '%s_%s'%(evid, event_name)
        ot = ot_dict[evid]
        lat, lon, dep, mag = [float(code) for code in codes[1:5]]
        if lat_min<=lat<=lat_max and lon_min<=lon<=lon_max and ot_min<=ot<=ot_max: to_add=True
        else: to_add = False; continue
        fout.write('{},{},{},{},{},{}\n'.format(id_name,ot,lat,lon,dep,mag))
    else: 
        if to_add: fout.write(line)
fout.close()
