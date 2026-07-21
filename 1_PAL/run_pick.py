"""Run PAL picker on locally stored daily waveform files."""

import argparse
import os
import warnings

from obspy import UTCDateTime

import config
import picker_pal


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/Example_data")
    parser.add_argument("--time_range", type=str, default="20190704-20190707")
    parser.add_argument("--sta_file", type=str, default="input/example_pal_format1.sta")
    parser.add_argument("--out_pick_dir", type=str, default="output/eg/picks")
    args = parser.parse_args()

    cfg = config.Config()
    get_data_dict = cfg.get_data_dict
    read_data = cfg.read_data
    sta_dict = cfg.get_sta_dict(args.sta_file)
    picker = picker_pal.STA_LTA_Kurtosis(
        win_sta=cfg.win_sta,
        win_lta=cfg.win_lta,
        trig_thres=cfg.trig_thres,
        p_win=cfg.p_win,
        s_win=cfg.s_win,
        pca_win=cfg.pca_win,
        pca_range=cfg.pca_range,
        fd_thres=cfg.fd_thres,
        amp_ratio_thres=cfg.amp_ratio_thres,
        amp_win=cfg.amp_win,
        win_kurt=cfg.win_kurt,
        det_gap=cfg.det_gap,
        to_prep=cfg.to_prep,
        freq_band=cfg.freq_band,
        vp=cfg.vp,
        vs=cfg.vs,
    )

    os.makedirs(args.out_pick_dir, exist_ok=True)
    start_date, end_date = [
        UTCDateTime(date) for date in args.time_range.split("-")
    ]
    num_days = (end_date.date - start_date.date).days
    print("run pick: raw_waveform --> picks")
    print("time range: {} to {}".format(start_date.date, end_date.date))
    for day_idx in range(num_days):
        date = start_date + day_idx * 86400
        data_dict = get_data_dict(date, args.data_dir)
        data_dict = {
            net_sta: paths for net_sta, paths in data_dict.items()
            if net_sta in sta_dict
        }
        pick_path = os.path.join(args.out_pick_dir, "{}.pick".format(date.date))
        with open(pick_path, "w") as out_pick:
            for net_sta, data_paths in sorted(data_dict.items()):
                print("-" * 40)
                print("picking {} {}".format(net_sta, date.date))
                stream = read_data(data_paths, sta_dict)
                picker.pick(stream, out_pick)


if __name__ == "__main__":
    main()
