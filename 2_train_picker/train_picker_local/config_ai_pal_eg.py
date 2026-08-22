"""Shared AI-PAL preprocessing, data-pipeline, and association parameters."""
import numpy as np
import data_pipeline


class Config(object):
  def __init__(self):
    # Shared waveform preprocessing and sample layout for AI models.
    self.samp_rate = 100
    self.win_len = 25
    self.win_stride = 2.5
    self.num_chn = 3
    self.freq_band = [1, 20]
    self.global_max_norm = False
    self.to_prep = True
    self.train_ratio = 0.9
    self.valid_ratio = 0.1
    self.max_assoc_ratio = 0.5
    # "phase" reads per-pick num_aug tags written by step 0; "fixed" retains
    # the legacy global augmentation count below.
    self.positive_num_aug_mode = "phase"
    self.num_aug = 2
    self.max_noise = 0.5
    self.p_context_sec = 0.5

    # Rarity-aware positive augmentation (FMD, hypocentral distance, and
    # spatiotemporal seismicity rate).
    self.rarity_augmentation_values = [1, 2, 3, 4]
    self.rarity_percentiles = [50, 75, 90]
    self.rarity_max_hypo_dist_km = 200.0
    self.rarity_mag_bin_width = 0.1
    self.rarity_hypo_dist_bin_width = 10.0
    self.rarity_spatial_bin_km = 5.0
    self.rarity_time_bin_days = 10.0

    # Shared continuous-picking controls.
    self.picker_batch_size = 512
    self.tp_dev = 1.0
    self.ts_dev = 1.5
    self.picker_min_cluster_size = 2
    self.data_buffer_sec = 60.0
    self.taper_max_length_sec = 10.0
    self.normalize_to_three_channels = True  # cycle/truncate available channels into E/N/Z

    # Cross-picker P/S-pair consensus. A value of 1 keeps picks detected by
    # any enabled picker; increase to 2+ for stricter ensemble agreement.
    self.ensemble_min_cluster_size = 1
    self.save_individual_picker_outputs = True

    # Shared data pipeline.
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

    # Shared amplitude measurement and waveform QC.
    self.amp_win = [1, 6]
    self.rm_glitch = True
    self.win_peak = 1
    self.amp_ratio_thres = [5, 9, 3]

    # PAL association parameters. The full station list is always required for
    # picking. With no optional subnet files, associate it under the "full" key.
    # Subnet entries inherit unspecified values from "default".
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
    }

    # P/S velocity ratio used to estimate station origin times from picks.
    self.vp = 5.9
    self.vs = 3.5
    # Halo around offline association intervals. This is not the rule-based
    # PAL picker S-arrival search window (`s_win`).
    self.association_buffer_sec = 30.0

    # Cross-subnetwork duplicate-event merging defaults. These values are kept
    # for consistent configs but are not used when association receives only
    # the single "full" station file.
    self.merge_origin_time_tol_sec = 2.5
    self.merge_epicenter_tol_km = 5.0
    self.merge_depth_tol_km = 10.0
    self.merge_min_shared_phase_stations = 4
    self.merge_phase_pick_time_tol_sec = 1.0
    self.merge_time_format_digits = 6
