#!/usr/bin/env python3
"""Associate existing PAL picks for one station file and date range."""

import os
import sys
from datetime import timedelta
from pathlib import Path

from data_pipeline_aws import to_associator_sta_dict
from run_pick_aws import parse_time_range


def geometry_key(sta_dict):
    return tuple(
        sorted((net_sta, row[0], row[1], row[2]) for net_sta, row in sta_dict.items())
    )


def run_assoc(
    run_time_range, station_file, input_pick_dir,
    output_catalog, output_phase, pal_source_dir, cfg,
):
    sys.path.insert(0, str(Path(pal_source_dir).expanduser().resolve()))
    import associator_pal

    start, end = parse_time_range(run_time_range)
    output_catalog = Path(output_catalog)
    output_phase = Path(output_phase)
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_phase.parent.mkdir(parents=True, exist_ok=True)
    ctlg_partial = output_catalog.with_suffix(output_catalog.suffix + ".partial")
    pha_partial = output_phase.with_suffix(output_phase.suffix + ".partial")
    current_geometry = None
    associator = None

    with ctlg_partial.open("w") as ctlg_fp, pha_partial.open("w") as pha_fp:
        current = start
        while current < end:
            active = cfg.get_sta_dict(station_file, current)
            picks = cfg.get_picks(current, input_pick_dir)
            if active and len(picks):
                picks = picks[[net_sta in active for net_sta in picks["net_sta"]]]
            if active and len(picks):
                pal_stations = to_associator_sta_dict(active)
                key = geometry_key(pal_stations)
                if key != current_geometry:
                    associator = associator_pal.PS_Pair_Assoc(
                        pal_stations,
                        xy_margin=cfg.xy_margin,
                        xy_grid=cfg.xy_grid,
                        z_grids=cfg.z_grids,
                        min_sta=cfg.min_sta,
                        ot_dev=cfg.ot_dev,
                        max_res=cfg.max_res,
                        max_drop=cfg.max_drop,
                        vp=cfg.vp,
                    )
                    current_geometry = key
                associator.associate(picks, ctlg_fp, pha_fp)
            print("assoc {}: {} stations, {} picks".format(
                current, len(active), len(picks),
            ))
            current += timedelta(days=1)

    os.replace(ctlg_partial, output_catalog)
    os.replace(pha_partial, output_phase)
