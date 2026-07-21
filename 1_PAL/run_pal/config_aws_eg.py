"""PAL picker and associator model parameters for an example SCEDC AWS run."""

import numpy as np
import data_pipeline_aws as dp


class Config(object):
  def __init__(self):
    # 1. picker params
    self.win_sta    = [0.8,0.4,1.]
    self.win_lta    = [6.,2.,2.]
    self.win_kurt   = [5.,1.]
    self.trig_thres = 12.
    self.p_win      = [.5,1.]
    self.s_win      = 10.
    self.pca_win    = 1.
    self.pca_range  = [0.,2.]
    self.fd_thres   = 2.5
    self.amp_ratio_thres = [5,8,3]
    self.amp_win    = [1.,5.]
    self.det_gap    = 5.
    self.to_prep    = True
    self.freq_band  = [1,20]

    # 2. associator params
    self.min_sta   = 4
    self.ot_dev    = 2.
    self.max_res   = 1.5
    self.max_drop  = 1
    self.xy_margin = 0.1
    self.xy_grid   = 0.02
    self.z_grids   = np.arange(2,20,3)
    self.vp        = 5.9
    self.vs        = 3.45

    # 3. data pipeline
    self.get_data_dict = dp.get_data_dict_aws
    self.get_sta_dict  = dp.get_sta_dict_aws
    self.get_picks     = dp.get_pal_picks
    self.read_data     = dp.read_data_aws
