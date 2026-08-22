"""Shared AI-PAL preprocessing, data-pipeline, and association parameters."""
import numpy as np

import data_pipeline


class Config(object):
  def __init__(self):
    # 1. Waveform preprocessing
    self.samp_rate = 100
    self.win_len = 25
    self.win_stride = 2.5
    self.num_chn = 3
    self.freq_band = [1, 20]
    self.global_max_norm = False
    self.to_prep = True
    self.p_context_sec = 0.5
    self.data_buffer_sec = 60.0
    self.taper_max_length_sec = 10.0
    self.normalize_to_three_channels = True
    self.location_priority = ["10", "20", "01", "00", ""]

    # 2. Continuous picking and picker ensemble
    self.picker_pos_neg_group = ["SAR", "PHN"]
    self.picker_pos_group = []  # Optional for continuous picking.
    self.picker_ref_group = ["PHN-SB"]
    self.picker_batch_size = 512
    self.tp_dev = 1.0
    self.ts_dev = 1.5
    self.picker_min_cluster_size = 2
    self.picker_pos_neg_group_min_picker_support = 1
    self.picker_pos_group_min_picker_support = 2
    self.save_individual_picker_outputs = True
    self.amp_win = [1, 6]
    self.rm_glitch = True
    self.win_peak = 1
    self.amp_ratio_thres = [6, 10, 3]

    # 3. Initial PAL association and subnet merge
    self.vp = 5.9
    self.vs = 3.5
    self.association_buffer_sec = 20.0
    self.association_interval_sec = 3600.0
    self.subnet_assoc_params = {
        "default": {
            "min_sta": 4,
            "ot_dev": 1.4,
            "max_res": 1.2,
            "max_drop": 1,
            "xy_margin": 0.1,
            "xy_grid": 0.02,
            "z_grids": np.arange(2, 25, 3),
            "vp": 5.9,
        },
        "full": {"min_sta": 4, "ot_dev": 1.4, "max_res": 1.2},
        "r1": {"min_sta": 4, "ot_dev": 1.4, "max_res": 1.2},
        "r2": {"min_sta": 4, "ot_dev": 1.4, "max_res": 1.2},
        "r3": {"min_sta": 4, "ot_dev": 1.4, "max_res": 1.2},
        "r4": {"min_sta": 4, "ot_dev": 1.4, "max_res": 1.2},
        "r5": {"min_sta": 4, "ot_dev": 1.4, "max_res": 1.2},
    }
    self.merge_origin_time_tol_sec = 2.5
    self.merge_epicenter_tol_km = 5.0
    self.merge_depth_tol_km = 10.0
    self.merge_min_shared_phase_stations = 4
    self.merge_phase_pick_time_tol_sec = 1.0
    self.merge_time_format_digits = 6

    # 4. Post-processing: event repicking and PAL reassociation
    self.enable_post_process = True
    self.repicker_pos_neg_group = ["SAR", "FT", "PHN", "RUN"]
    self.repicker_pos_group = ["SAR", "FT", "PHN", "RUN"]
    self.repick_phase_buffer_sec = 2.0
    self.repick_num_repeat = 20
    self.repick_batch_size = 128
    self.repick_random_seed = 20250708
    self.repick_min_window_vote_ratio = 0.2
    self.repick_group_min_picker_support = 2

    # 5. Final event products
    self.enable_event_waveform_plot = True
    self.enable_event_waveform_plot_ref = [0]
    self.save_filtered_event_waveforms = False

    # 6. Training-sample construction
    self.train_ratio = 0.9
    self.valid_ratio = 0.1
    self.max_assoc_ratio = 0.5
    self.positive_num_aug_mode = "phase"
    self.num_aug = 2
    self.max_noise = 0.5
    self.rarity_augmentation_values = [1, 2, 3, 4]
    self.rarity_percentiles = [50, 75, 90]
    self.rarity_max_hypo_dist_km = 200.0
    self.rarity_mag_bin_width = 0.1
    self.rarity_hypo_dist_bin_width = 10.0
    self.rarity_spatial_bin_km = 5.0
    self.rarity_time_bin_days = 10.0

    # 7. Data-pipeline bindings
    self.read_fpha = data_pipeline.read_fpha
    self.read_fpick = data_pipeline.read_fpick
    self.read_assoc_rate = data_pipeline.read_assoc_rate
    self.get_data_dict = data_pipeline.get_data_dict
    self.get_buffered_data_dict = data_pipeline.get_buffered_data_dict
    self.load_station_stream = data_pipeline.load_station_stream
    self.get_sta_dict = data_pipeline.get_sta_dict
    self.read_data = data_pipeline.read_data
    self.get_picks = data_pipeline.get_picks
    self.dtime2str = data_pipeline.dtime2str
