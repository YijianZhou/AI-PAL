#!/usr/bin/env python3
"""Run PAL picking directly from the public SCEDC S3 archive."""

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from data_pipeline_aws import build_s3_client



def parse_date(value):
    return datetime.strptime(value, "%Y%m%d").date()


def parse_time_range(value):
    start_text, end_text = value.split("-", 1)
    start, end = parse_date(start_text), parse_date(end_text)
    if start >= end:
        raise ValueError("time_range must have start < exclusive end")
    return start, end


def build_picker(pal_source_dir, cfg):
    sys.path.insert(0, str(Path(pal_source_dir).expanduser().resolve()))
    import picker_pal

    picker = picker_pal.STA_LTA_Kurtosis(
        win_sta=cfg.win_sta, win_lta=cfg.win_lta,
        trig_thres=cfg.trig_thres, p_win=cfg.p_win, s_win=cfg.s_win,
        pca_win=cfg.pca_win, pca_range=cfg.pca_range,
        fd_thres=cfg.fd_thres, amp_ratio_thres=cfg.amp_ratio_thres,
        amp_win=cfg.amp_win, win_kurt=cfg.win_kurt, det_gap=cfg.det_gap,
        to_prep=cfg.to_prep, freq_band=cfg.freq_band,
        vp=cfg.vp, vs=cfg.vs,
    )
    return picker, cfg


def process_day(
    observed_date, station_file, pick_dir, picker, cfg, s3_client,
    s3_bucket, s3_root_prefix, loc_priority, acceleration_codes,
    to_overwrite=False, retry_failed=False,
):
    pick_dir = Path(pick_dir)
    status_dir = pick_dir.parent / "pick_status"
    pick_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    stem = observed_date.isoformat()
    pick_path = pick_dir / (stem + ".pick")
    done_path = status_dir / (stem + ".done.json")
    failed_path = status_dir / (stem + ".failed.json")
    partial_path = pick_path.with_suffix(".pick.partial")
    if not to_overwrite:
        if done_path.exists():
            print("skip completed day {}".format(stem))
            return
        if failed_path.exists() and not retry_failed:
            print("skip accepted partial day {}".format(stem))
            return

    active = cfg.get_sta_dict(station_file, observed_date)
    data_dict = cfg.get_data_dict(
        observed_date, active, s3_client,
        bucket=s3_bucket, root_prefix=s3_root_prefix,
        location_priority=loc_priority,
    )
    errors = []
    num_processed = 0
    with partial_path.open("w", encoding="utf-8") as out_pick:
        for index, (net_sta, records) in enumerate(sorted(data_dict.items()), start=1):
            print("{} {}/{}: {} {} ({} object(s))".format(
                stem, index, len(data_dict), net_sta,
                active[net_sta]["band"], len(records),
            ))
            try:
                stream = cfg.read_data(
                    records, active[net_sta], s3_client, bucket=s3_bucket,
                    acceleration_instrument_codes=acceleration_codes,
                )
                picker.pick(stream, out_pick)
                num_processed += 1
            except Exception as exc:
                errors.append({
                    "net_sta": net_sta,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                print("ERROR {}: {}".format(net_sta, exc), file=sys.stderr)

    os.replace(partial_path, pick_path)
    status = {
        "date": stem,
        "active_station_epochs": len(active),
        "stations_with_usable_s3_components": len(data_dict),
        "stations_processed": num_processed,
        "station_errors": errors,
        "pick_file": str(pick_path),
    }
    status_path = failed_path if errors else done_path
    stale_status_path = done_path if errors else failed_path
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if stale_status_path.exists():
        stale_status_path.unlink()
    result = "failed" if errors else "completed"
    print("{} {}: {}/{} stations, {} errors".format(
        result, stem, num_processed, len(data_dict), len(errors),
    ))


def run_pick(
    run_time_range, station_file, pick_dir, pal_source_dir, cfg,
    s3_bucket, s3_region, s3_root_prefix, s3_access_mode,
    loc_priority, acceleration_codes, to_overwrite, retry_failed=False,
):
    station_file = Path(station_file)
    if not station_file.exists():
        raise FileNotFoundError(station_file)
    if not Path(pal_source_dir).exists():
        raise FileNotFoundError("PAL source directory not found: {}".format(pal_source_dir))

    start, end = parse_time_range(run_time_range)
    picker, cfg = build_picker(pal_source_dir, cfg)
    s3_client = build_s3_client(s3_region, s3_access_mode)
    current = start
    while current < end:
        process_day(
            current, station_file, pick_dir, picker, cfg, s3_client,
            s3_bucket, s3_root_prefix, loc_priority, acceleration_codes,
            to_overwrite, retry_failed,
        )
        current += timedelta(days=1)
