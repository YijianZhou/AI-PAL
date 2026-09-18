"""Shared AI-PAL preprocessing, data-pipeline, and association parameters."""
import numpy as np

import data_pipeline


class Config(object):
  def __init__(self):
    # 0. Console output
    self.console_verbosity = "default"  # quiet, default, or debug

    # 1. Waveform preprocessing
    self.samp_rate = 100
    self.win_len = 25
    self.win_stride = 2.5
    self.num_chn = 3
    self.freq_band = [1, 20]
    self.global_max_norm = False
    self.waveform_backend = "scedc"  # "local" archive or "scedc" S3; tune when copying.
    self.to_prep = True  # True for raw traces; False for prepared local archives.
    self.to_filter = True  # apply freq_band before AI inference
    self.channel_priority = ["HH", "BH", "EH", "HN", "EN", "SH"]
    self.p_context_sec = 0.5
    self.data_buffer_sec = 60.0
    self.taper_max_length_sec = 10.0
    self.normalize_to_three_channels = True
    self.location_priority = ["10", "20", "01", "00", ""]

    # 2. Continuous picking and picker ensemble
    self.picker_pos_neg_group = ["SAR", "PHN"]
    self.picker_batch_size = 512
    self.tp_dev = 1.0
    self.ts_dev = 1.5
    self.picker_min_cluster_size = 2
    self.picker_pos_neg_group_min_picker_support = 1
    self.save_individual_picker_outputs = True

    # Final pick quality (0 best, 3 fallback); does not reject picks.
    self.pick_quality_both_groups_code = 0  # POS_NEG and POS agree.
    self.pick_quality_strong_vote_ratio = 0.5  # Strictly greater than this ratio.
    self.pick_quality_code1_min_pickers = 2  # Strong pickers needed for code 1.
    self.pick_quality_code2_min_pickers = 1  # Otherwise code 2; fewer gives 3.
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
            "ot_dev": 1.2,
            "max_res": 1.0,
            "max_drop": 1,
            "xy_margin": 0.1,
            "xy_grid": 0.02,
            "z_grids": np.arange(2, 25, 3),
            "vp": 5.9,
        },
        "full": {"min_sta": 4, "ot_dev": 1.2, "max_res": 1.0},
        "r1": {"min_sta": 4, "ot_dev": 1.2, "max_res": 1.0},
        "r2": {"min_sta": 4, "ot_dev": 1.2, "max_res": 1.0},
        "r3": {"min_sta": 4, "ot_dev": 1.0, "max_res": 0.8},
        "r4": {"min_sta": 4, "ot_dev": 1.2, "max_res": 1.0},
        "r5": {"min_sta": 4, "ot_dev": 1.0, "max_res": 0.8},
        "r6": {"min_sta": 4, "ot_dev": 1.5, "max_res": 1.2},
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
    self.enable_event_waveform_plot = False
    self.save_filtered_event_waveforms = False

    # 6. Training-sample construction
    self.train_ratio = 0.9
    self.valid_ratio = 0.1
    self.max_assoc_ratio = 0.5
    self.positive_num_aug_mode = "phase"  # "phase": per-pick num_aug tags; "fixed": global num_aug.
    self.num_aug = None  # Unused in "phase" mode; set >= 1 for "fixed" mode.
    self.max_noise = 0.5
    self.rarity_augmentation_values = [1, 2, 3, 4]
    self.rarity_percentiles = [50, 75, 90]
    self.rarity_max_hypo_dist_km = 200.0
    self.rarity_mag_bin_width = 0.1
    self.rarity_hypo_dist_bin_width = 10.0
    self.rarity_spatial_bin_km = 5.0
    self.rarity_time_bin_days = 10.0

    # 7. Data-pipeline bindings
    waveform_pipeline = data_pipeline
    station_loader = data_pipeline.load_station_stream
    if self.waveform_backend == "scedc":
        import data_pipeline_ai_aws as waveform_pipeline
        station_loader = waveform_pipeline.load_station_stream
    elif self.waveform_backend == "scedc_training":
        import data_pipeline_training_aws as training_pipeline
        station_loader = training_pipeline.load_station_stream
    elif self.waveform_backend != "local":
        raise ValueError("waveform_backend must be local, scedc, or scedc_training")
    self.read_fpha = data_pipeline.read_fpha
    self.read_fpick = data_pipeline.read_fpick
    self.read_assoc_rate = data_pipeline.read_assoc_rate
    self.get_data_dict = waveform_pipeline.get_data_dict
    self.get_buffered_data_dict = waveform_pipeline.get_buffered_data_dict
    self.load_station_stream = station_loader
    self.get_sta_dict = waveform_pipeline.get_sta_dict
    self.read_data = waveform_pipeline.read_data
    self.get_picks = data_pipeline.get_picks
    self.dtime2str = data_pipeline.dtime2str
