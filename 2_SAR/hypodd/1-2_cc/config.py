""" Configure file for hypoDD interface
    Download hypoDD at https://www.ldeo.columbia.edu/~felixw/hypoDD.html
"""

class Config(object):
  def __init__(self):

    # 1. i/o paths
    self.hypo_root = '/home/zhouyj/bin'
    self.ctlg_code = 'eg_sar_cc'
    self.fsta = 'input/station_eg.csv'
    self.fpha_name = 'input/eg_sar.pha'
    self.fpha_ot = 'input/eg_sar_hyp_full.pha'
    self.fpha_loc = 'input/eg_sar_ct_full.pha'
    self.out_ctlg = 'output/%s.ctlg'%self.ctlg_code
    self.out_pha = 'output/%s.pha'%self.ctlg_code
    self.out_pha_full = 'output/%s_full.pha'%self.ctlg_code
    self.event_root = '/data/bigdata/zhouyj/Example_events'
    self.data_dir = '/data/Example_data'
    # 2. ph2dt_cc
    # 2.1 event linkage (initial calc & further selection)
    self.cc_thres = [0.3, 0.3]             # CC thres for event pair
    self.loc_dev_thres = [3, 3]            # km, maximum x-y location separation
    self.dep_dev_thres = [4, 4]            # km, maximum depth separation
    self.dist_thres = [100, 80]            # km, max epicentral dist
    self.dt_thres = [[0.6,1.], [0.5,0.8]]  # sec, max P & S dtime
    self.num_sta_thres = [4,4]             # min sta_num for one event pair
    self.max_sta = 15                      # max sta_num for one event pair
    self.max_nbr = 100                     # max number of neighbor event
    self.temp_mag = 0.                     # min mag for templates
    self.temp_sta = 4                      # min sta_num for templates
    # 2.2 data preprocess
    self.freq_band = [1.,16.]
    self.samp_rate = 100
    self.chn_p = [2]                       # chn for P picking
    self.chn_s = [0,1]                     # chn for S picking
    self.win_temp_p = [0.5,2.]
    self.win_temp_s = [0.2,3.8]            # pre-post phase arrival
    self.win_event = [5, 20]               # event data cutting, just long enough
    # 3. run hypoDD
    self.ot_range = '20190704-20190710'
    self.lat_range = [35.45,36.05]
    self.lon_range = [-117.8,-117.25]
    self.num_grids = [1,1]    # x,y (lon, lat)
    self.xy_pad = [0.045,0.036]    # degree, additional events for reloc on one grid
    self.dep_corr = 5    # avoid air quake, modify velo_mod accordingly
    self.num_workers = 10
    self.keep_grids = False
