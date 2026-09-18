"""Event-scoped offline positive repicking, PAL reassociation, and products."""

import gc
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
from obspy import UTCDateTime

from association_runner import (
    load_station_geometry, merge_canonical_interval, parse_date_range,
    processing_day_bounds,
)
from event_repicker import EVENT_REPICK_VERSION, EventRepicker
from phase_merge import group_events, is_event_header, read_phase_file
from picker_stream import RetainedStationWaveform, configure_torch_backends
from data_pipeline import preprocess_picker_stream
import runtime_console


PROGRESS_EVERY_EVENTS = 1000


def _count_phase_events(path):
    count = 0
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            codes = [value.strip() for value in line.strip().split(",")]
            if is_event_header(codes):
                count += 1
    return count


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def _print_progress(num_done, num_all, started, final=False):
    elapsed = max(0.0, time.perf_counter() - started)
    rate = num_done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, num_all - num_done)
    eta = remaining / rate if rate > 0 else 0.0
    label = "final" if final else "checkpoint"
    runtime_console.log(
        "progress",
        "postprocess {} | {}/{} initial events | elapsed {} | "
        "{:.2f} events/s | ETA {}".format(
            label,
            num_done,
            num_all,
            _format_duration(elapsed),
            rate,
            _format_duration(eta),
        ),
    )


def _station_paths(cfg, data_dir, station, start_time, end_time):
    base = ".".join(str(station).split(".")[:2])
    paths = []
    day = UTCDateTime(str(UTCDateTime(start_time).date))
    last_day = UTCDateTime(str(UTCDateTime(end_time).date))
    while day <= last_day:
        daily = cfg.get_data_dict(
            day,
            str(data_dir),
            normalize_to_three_channels=bool(getattr(
                cfg, "normalize_to_three_channels", True
            )),
        )
        paths.extend(daily.get(base, []))
        day += 86400
    return list(dict.fromkeys(paths))


def _read_preprocessed_station(
    station, bounds, cfg, data_dir, stations,
):
    requested_start, requested_end = [UTCDateTime(value) for value in bounds]
    taper_sec = float(cfg.taper_max_length_sec)
    raw_start = requested_start - taper_sec
    raw_end = requested_end + taper_sec
    paths = _station_paths(
        cfg, data_dir, station, raw_start, raw_end
    )
    if not paths:
        print("no event waveform files for {}".format(station), flush=True)
        return station, None
    stream = cfg.read_data(
        paths,
        stations,
        start_time=raw_start,
        end_time=raw_end,
        normalize_to_three_channels=bool(getattr(
            cfg, "normalize_to_three_channels", True
        )),
        to_prep=bool(getattr(cfg, "to_prep", True)),
        location_priority=getattr(
            cfg, "location_priority", ("10", "20", "01", "02", "00", "")
        ),
        channel_priority=getattr(
            cfg, "channel_priority", ("HH", "BH", "EH", "HN", "EN", "SH")
        ),
    )
    if not stream:
        return station, None
    filtered, _ = preprocess_picker_stream(
        stream,
        num_channels=int(cfg.num_chn),
        sampling_rate=float(cfg.samp_rate),
        min_length_sec=float(cfg.win_len),
        frequency_band=cfg.freq_band,
        taper_max_length_sec=taper_sec,
        to_filter=bool(getattr(cfg, "to_filter", True)),
        retain_raw=False,
    )
    if len(filtered) != int(cfg.num_chn):
        return station, None
    filtered = filtered.slice(
        requested_start, requested_end, nearest_sample=True
    ).copy()
    if len(filtered) != int(cfg.num_chn):
        return station, None
    for trace in filtered:
        trace.data = np.asarray(trace.data, dtype=np.float32)
    return station, RetainedStationWaveform(station, filtered)


def _load_waveform_context(
    repicker, events, cfg, data_dir, stations, num_workers,
):
    requests = repicker.waveform_requests(events)
    if not requests:
        return [], 0
    tasks = sorted(requests.items())

    def process(item):
        return _read_preprocessed_station(
            item[0], item[1], cfg, data_dir, stations
        )

    if num_workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(
            max_workers=min(num_workers, len(tasks)),
            thread_name_prefix="event-waveform",
        ) as executor:
            loaded = list(executor.map(process, tasks))
    else:
        loaded = [process(item) for item in tasks]
    waveforms = {
        station: holder for station, holder in loaded if holder is not None
    }
    context = [{
        "start": min(bounds[0] for bounds in requests.values()),
        "end": max(bounds[1] for bounds in requests.values()),
        "waveforms": waveforms,
    }]
    return context, len(requests)


def _release_context(context):
    for entry in context:
        for holder in entry.get("waveforms", {}).values():
            holder.release()
        entry.get("waveforms", {}).clear()
    gc.collect()


def _snapshot_matches_event(snapshot, event, cfg):
    groups = group_events(
        [snapshot["event"], event],
        cfg.merge_origin_time_tol_sec,
        cfg.merge_epicenter_tol_km,
        cfg.merge_depth_tol_km,
        cfg.merge_min_shared_phase_stations,
        cfg.merge_phase_pick_time_tol_sec,
    )
    return len(groups) == 1


def _release_snapshots(snapshots):
    for snapshot in snapshots:
        for holder in snapshot.get("waveforms", {}).values():
            holder.release()
        snapshot.get("waveforms", {}).clear()
    gc.collect()


def run_offline_event_postprocessing(
    ai_pal_root, cfg, repicker_pos_neg_specs, repicker_pos_specs,
    data_dir, station_file,
    initial_phase_dir, final_root, time_range, num_workers=1,
    overwrite=False, day_complete_callback=None,
):
    """Repick/reassociate initial detections one event at a time."""
    runtime_console.configure(cfg)
    if not bool(getattr(cfg, "enable_post_process", False)):
        raise ValueError(
            "2.3 postprocessing requires enable_post_process=True"
        )
    if not repicker_pos_neg_specs or not repicker_pos_specs:
        raise ValueError("both event repicker groups must be selected")
    cfg.repick_num_workers = max(1, int(num_workers))
    configure_torch_backends(cfg)

    data_dir = Path(data_dir)
    station_file = Path(station_file)
    initial_phase_dir = Path(initial_phase_dir)
    final_root = Path(final_root)
    status_root = final_root / "postprocess_status"
    plot_root = final_root / "event_waveform"
    waveform_root = final_root / "event_waveforms"
    internal_root = final_root / "_internal" / "event_postprocess"
    for path in (final_root, status_root, internal_root):
        path.mkdir(parents=True, exist_ok=True)
    cfg.out_root = str(final_root)

    start_date, end_date = parse_date_range(time_range)
    target_dates = [
        start_date + timedelta(days=index)
        for index in range((end_date - start_date).days)
    ]
    repicker = EventRepicker(
        ai_pal_root, cfg, repicker_pos_neg_specs, station_file,
        station_dict=load_station_geometry(cfg, station_file, start_date),
        repicker_pos_specs=repicker_pos_specs,
    )
    repicker.load_pickers()
    initial_phase_paths = {
        date: initial_phase_dir / "phase_{}.dat".format(date.isoformat())
        for date in target_dates
    }
    missing_initial = [
        path for path in initial_phase_paths.values() if not path.exists()
    ]
    if missing_initial:
        raise FileNotFoundError(missing_initial[0])
    num_all_events = sum(
        _count_phase_events(path) for path in initial_phase_paths.values()
    )
    runtime_console.log(
        "AI-PAL",
        "offline postprocess | days={} | initial_events={} | workers={}".format(
            len(target_dates), num_all_events, max(1, int(num_workers))
        ),
    )
    num_done_events = 0
    next_progress_event = PROGRESS_EVERY_EVENTS
    progress_started = time.perf_counter()
    print(
        "postprocessing target: {} initial events | progress every {} events"
        .format(num_all_events, PROGRESS_EVERY_EVENTS),
        flush=True,
    )
    daily_phase_paths = []
    unreleased_snapshots = []
    try:
        for current_date in target_dates:
            current_stations = load_station_geometry(
                cfg, station_file, current_date
            )
            repicker.set_station_geometry(current_stations)
            # Local files may use NET.STA while static station dictionaries
            # use NET.STA.BAND. The AWS dictionary is already NET.STA.
            stations = dict(current_stations)
            for selector, metadata in list(current_stations.items()):
                stations.setdefault(
                    ".".join(selector.split(".")[:2]), metadata
                )
            initial_phase = initial_phase_paths[current_date]
            initial_events = read_phase_file(initial_phase)
            day_start, day_end = processing_day_bounds(cfg, current_date)
            daily_phase = final_root / (
                "phase_{}.dat".format(current_date.isoformat())
            )
            status_path = status_root / (
                "{}.json".format(current_date.isoformat())
            )
            daily_phase_paths.append(daily_phase)
            print(
                "postprocessing {}: {} initial events".format(
                    current_date, len(initial_events)
                ),
                flush=True,
            )
            runtime_console.log(
                "day",
                "{} postprocess start | {} initial events".format(
                    current_date, len(initial_events)
                ),
            )
            completed_version = None
            if status_path.exists():
                try:
                    completed_version = json.loads(
                        status_path.read_text(encoding="utf-8")
                    ).get("repick_version")
                except (OSError, ValueError, TypeError):
                    completed_version = None
            if (
                not overwrite and daily_phase.exists()
                and completed_version == EVENT_REPICK_VERSION
            ):
                num_done_events += len(initial_events)
                if num_done_events >= next_progress_event:
                    _print_progress(
                        num_done_events, num_all_events, progress_started
                    )
                    next_progress_event = (
                        num_done_events // PROGRESS_EVERY_EVENTS + 1
                    ) * PROGRESS_EVERY_EVENTS
                print("skip completed postprocess day: {}".format(daily_phase))
                continue

            reassociated_events = []
            waveform_snapshots = []
            day_summary = {
                "date": current_date.isoformat(),
                "repick_version": EVENT_REPICK_VERSION,
                "num_initial_events": len(initial_events),
                "num_station_event_attempts": 0,
                "num_repicker_phase_pairs_generated": 0,
                "num_events_reassociated": 0,
                "num_events_reassociation_rejected": 0,
                "num_phase_pairs_both_groups": 0,
                "num_phase_pairs_pos_neg_only": 0,
                "num_phase_pairs_pos_only": 0,
            }
            with tempfile.TemporaryDirectory(
                prefix="{}_".format(current_date.isoformat()),
                dir=internal_root,
            ) as temp_dir:
                temp_dir = Path(temp_dir)
                work_phase = temp_dir / "event_phase.dat"
                work_catalog = temp_dir / "event_catalog.dat"
                for event in initial_events:
                    repicker.write_events([event], work_phase, work_catalog)
                    context = []
                    try:
                        context, _ = _load_waveform_context(
                            repicker,
                            [event],
                            cfg,
                            data_dir,
                            stations,
                            max(1, int(num_workers)),
                        )
                        summary = repicker.process_event(
                            event["time"],
                            work_phase,
                            work_catalog,
                            context,
                            {},
                        )
                        event_outputs = read_phase_file(work_phase)
                        reassociated_events.extend(event_outputs)
                        for key in (
                            "num_station_event_attempts",
                            "num_repicker_phase_pairs_generated",
                            "num_events_reassociated",
                            "num_events_reassociation_rejected",
                            "num_phase_pairs_both_groups",
                            "num_phase_pairs_pos_neg_only",
                            "num_phase_pairs_pos_only",
                        ):
                            day_summary[key] += int(summary.get(key, 0))
                        if bool(getattr(
                            cfg, "enable_event_waveform_plot", False
                        )) or bool(getattr(
                            cfg, "save_filtered_event_waveforms", False
                        )):
                            new_snapshots = repicker.capture_event_waveforms(
                                event_outputs, context
                            )
                            waveform_snapshots.extend(new_snapshots)
                            unreleased_snapshots.extend(new_snapshots)
                    finally:
                        _release_context(context)

                    num_done_events += 1
                    if num_done_events >= next_progress_event:
                        _print_progress(
                            num_done_events,
                            num_all_events,
                            progress_started,
                        )
                        next_progress_event += PROGRESS_EVERY_EVENTS

                candidate_phase = temp_dir / "daily_candidates.dat"
                candidate_catalog = temp_dir / "daily_candidates_catalog.dat"
                groups_path = temp_dir / "daily_event_groups.csv"
                repicker.write_events(
                    reassociated_events, candidate_phase, candidate_catalog
                )
                merge_summary = merge_canonical_interval(
                    day_start,
                    day_end,
                    {"post_reassociation": candidate_phase},
                    daily_phase,
                    candidate_catalog,
                    groups_path,
                    cfg,
                )
                final_events = read_phase_file(daily_phase)
                day_summary["post_reassociation_merge"] = merge_summary
                day_summary["num_final_events"] = len(final_events)

                num_plots = 0
                waveform_summary = {}
                for final_event in final_events:
                    matched = [
                        snapshot for snapshot in waveform_snapshots
                        if _snapshot_matches_event(snapshot, final_event, cfg)
                    ]
                    if not matched:
                        continue
                    if bool(getattr(
                        cfg, "enable_event_waveform_plot", False
                    )):
                        num_plots += repicker.plot_events(
                            [final_event], matched, plot_root
                        )
                    if bool(getattr(
                        cfg, "save_filtered_event_waveforms", False
                    )):
                        result = repicker.save_filtered_event_waveforms(
                            [final_event], matched, waveform_root
                        )
                        for key, value in result.items():
                            waveform_summary[key] = (
                                waveform_summary.get(key, 0) + value
                            )
                day_summary["num_event_plots"] = num_plots
                day_summary.update(waveform_summary)
                _release_snapshots(waveform_snapshots)
                released_ids = {id(value) for value in waveform_snapshots}
                unreleased_snapshots[:] = [
                    value for value in unreleased_snapshots
                    if id(value) not in released_ids
                ]

            partial = status_path.with_suffix(".json.partial")
            partial.write_text(
                json.dumps(day_summary, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            partial.replace(status_path)
            if day_complete_callback is not None:
                day_complete_callback(
                    current_date, daily_phase, status_path, day_summary
                )
            print(
                "daily postprocess complete: {} | {} initial -> {} final "
                "events | {}".format(
                    current_date,
                    len(initial_events),
                    day_summary["num_final_events"],
                    daily_phase,
                ),
                flush=True,
            )
            runtime_console.log(
                "day",
                "day={} | {} initial events -> {} final events".format(
                    current_date,
                    len(initial_events),
                    day_summary["num_final_events"],
                ),
            )
    finally:
        _release_snapshots(unreleased_snapshots)
        repicker.close()

    _print_progress(
        num_done_events, num_all_events, progress_started, final=True
    )

    print(
        "daily postprocessed phase files: {}".format(len(daily_phase_paths)),
        flush=True,
    )
    runtime_console.log(
        "output",
        "postprocess complete | {} daily phase files | {}".format(
            len(daily_phase_paths), final_root
        ),
    )
    return daily_phase_paths
