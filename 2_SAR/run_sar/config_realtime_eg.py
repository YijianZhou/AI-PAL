"""Parameters for pseudo-realtime AI-PAL/SAR inference and association."""
import numpy as np


class Config(object):
  def __init__(self):
    # 1. SAR model and inference config
    self.samp_rate = 100
    self.win_len = 25
    self.win_stride = 10
    self.num_chn = 3
    self.freq_band = [1, 20]
    self.global_max_norm = False

    self.rnn_hidden_size = 128
    self.rnn_num_layers = 2
    self.rnn_step_len = 0.5
    self.rnn_step_stride = 0.1
    self.rnn_num_steps = int((self.win_len - self.rnn_step_len) / self.rnn_step_stride) + 1
    self.num_att_heads = 4

    self.trig_thres = 0.4
    self.picker_batch_size = 20
    self.tp_dev = 1.5
    self.ts_dev = 1.5
    self.amp_win = [1, 6]
    self.rm_glitch = True
    self.win_peak = 1
    self.amp_ratio_thres = [5, 10, 3]

    # Picker groups. The first group is the preferred AI-PAL family and the
    # following groups are reference pickers. Each enabled picker is run and
    # associated independently for now; ensemble logic will be added later.
    self.picker_groups = [["SAR"], ["PHN-SB"]]
    self.association_methods = ["PAL"]

    # SeisBench PhaseNet-specific inference parameters. Overlapping prediction
    # curves are averaged internally; resulting phase picks are merged with the
    # shared tp_dev/ts_dev before all forward P/S combinations are generated.
    self.phn_sb_weights = "original"
    self.phn_sb_p_threshold = 0.3
    self.phn_sb_s_threshold = 0.3
    self.phn_sb_overlap = 1500
    
    
    # 2. PAL associator config for each sub-network
    self.subnet_assoc_params = {
        "default": {
            "min_sta": 4, "ot_dev": 1.4, "max_res": 1.2, "max_drop": 1,
            "xy_margin": 0.1, "xy_grid": 0.02,
            "z_grids": np.arange(2, 20, 3), "vp": 5.9,
        },
        "r1": {
            "min_sta": 4, "ot_dev": 1.4, "max_res": 1.2, "max_drop": 1,
            "xy_margin": 0.1, "xy_grid": 0.02,
            "z_grids": np.arange(2, 20, 3), "vp": 5.9,
        },
        "r2": {
            "min_sta": 4, "ot_dev": 1.4, "max_res": 1.2, "max_drop": 1,
            "xy_margin": 0.1, "xy_grid": 0.02,
            "z_grids": np.arange(2, 20, 3), "vp": 5.9,
        },
        "r3": {
            "min_sta": 4, "ot_dev": 1.4, "max_res": 1.2, "max_drop": 1,
            "xy_margin": 0.1, "xy_grid": 0.02,
            "z_grids": np.arange(2, 20, 3), "vp": 5.9,
        },
        "r4": {
            "min_sta": 4, "ot_dev": 1.4, "max_res": 1.2, "max_drop": 1,
            "xy_margin": 0.1, "xy_grid": 0.02,
            "z_grids": np.arange(2, 20, 3), "vp": 5.9,
        },
        "r5": {
            "min_sta": 4, "ot_dev": 1.4, "max_res": 1.2, "max_drop": 1,
            "xy_margin": 0.1, "xy_grid": 0.02,
            "z_grids": np.arange(2, 20, 3), "vp": 5.9,
        },
    }

    # P/S velocity ratio used to estimate station origin times from picks.
    self.vp = 5.9
    self.vs = 3.5

    # 3. Merge params for duplicate detections across sub-networks
    self.merge_origin_time_tol_sec = 2.5
    self.merge_epicenter_tol_km = 5.0
    self.merge_depth_tol_km = 10.0
    self.merge_min_shared_phase_stations = 4
    self.merge_phase_pick_time_tol_sec = 1.0
    self.merge_time_format_digits = 6

    # 4. Realtime loop params. Set max_files/max_runtime_sec to 0 for no limit.
    self.poll_interval_sec = 10
    self.max_files = 0
    self.max_runtime_sec = 0

    # 5. Realtime waveform selection params
    # Prefer borehole location codes when the same NET.STA.CH group appears at
    # multiple location codes.  Adjust this list for SCSN conventions.
    self.location_priority = ["10", "20", "01", "00", ""]
