"""Run PAL associator on existing local pick files."""

import argparse
import os
import warnings

from obspy import UTCDateTime

import associator_pal
import config


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick_dir", type=str, default="output/eg/picks")
    parser.add_argument("--time_range", type=str, default="20171003-20171004")
    parser.add_argument("--sta_file", type=str, default="input/example_pal_format1.sta")
    parser.add_argument("--out_ctlg", type=str, default="output/eg/catalog.dat")
    parser.add_argument("--out_pha", type=str, default="output/eg/phase.dat")
    args = parser.parse_args()

    cfg = config.Config()
    get_picks = cfg.get_picks
    sta_dict = cfg.get_sta_dict(args.sta_file)
    associator = associator_pal.PS_Pair_Assoc(
        sta_dict,
        xy_margin=cfg.xy_margin,
        xy_grid=cfg.xy_grid,
        z_grids=cfg.z_grids,
        min_sta=cfg.min_sta,
        ot_dev=cfg.ot_dev,
        max_res=cfg.max_res,
        max_drop=cfg.max_drop,
        vp=cfg.vp,
    )

    out_root = os.path.split(args.out_ctlg)[0]
    if out_root:
        os.makedirs(out_root, exist_ok=True)
    start_date, end_date = [
        UTCDateTime(date) for date in args.time_range.split("-")
    ]
    num_days = (end_date.date - start_date.date).days
    print("run assoc: picks --> events")
    print("time range: {} to {}".format(start_date.date, end_date.date))
    with open(args.out_ctlg, "w") as out_ctlg, open(args.out_pha, "w") as out_pha:
        for day_idx in range(num_days):
            date = start_date + day_idx * 86400
            picks = get_picks(date, args.pick_dir)
            picks = picks[[net_sta in sta_dict for net_sta in picks["net_sta"]]]
            associator.associate(picks, out_ctlg, out_pha)


if __name__ == "__main__":
    main()
