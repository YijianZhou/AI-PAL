"""Daily offline picker inference with rolling hourly PAL association."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Lock

from obspy import UTCDateTime

from association_runner import (
    associate_subnet_picks,
    buffered_pick_interval_arrays,
    get_association_buffer_sec,
    load_station_geometry,
    merge_buffered_interval_candidates,
    merge_canonical_interval,
    parse_date_range,
    utc_day,
    write_association_rate_from_picks,
)
from offline_picker_runner import run_offline_picker_ensemble
from phase_merge import read_phase_file
import runtime_console


def _range_text(start, end):
    return "{}-{}".format(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))


def _interval_code(value):
    return UTCDateTime(value).strftime("%Y%m%dT%H%M%SZ")


def _combine_files(paths, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as output_fp:
        for path in paths:
            path = Path(path)
            if path.exists():
                output_fp.write(path.read_text(encoding="utf-8"))
    partial.replace(output_path)


class RollingHourlyAssociator(object):
    """Associate independent buffered hours concurrently."""

    def __init__(
        self, cfg, subnet_station_files, assoc_root, final_root,
        target_time_range, num_workers=1, hour_complete_callback=None,
    ):
        self.cfg = cfg
        self.subnet_station_files = {
            name: Path(path) for name, path in subnet_station_files.items()
        }
        self.assoc_root = Path(assoc_root)
        self.final_root = Path(final_root)
        self.target_start, self.target_end = parse_date_range(target_time_range)
        self.num_hour_workers = max(1, int(num_workers))
        self.buffer_seconds = get_association_buffer_sec(cfg)
        self.day_shift_seconds = float(getattr(cfg, "data_buffer_sec", 0.0))
        if self.day_shift_seconds < 0:
            raise ValueError("data_buffer_sec must be nonnegative")
        self.interval_seconds = float(
            getattr(cfg, "association_interval_sec", 3600.0)
        )
        if self.interval_seconds <= 0 or 86400 % self.interval_seconds != 0:
            raise ValueError(
                "association_interval_sec must divide one UTC day exactly"
            )
        if self.buffer_seconds > self.interval_seconds:
            raise ValueError(
                "association_buffer_sec cannot exceed association_interval_sec "
                "when only the final waveform interval is retained"
            )
        self.intervals_per_day = int(86400 / self.interval_seconds)
        self.hour_complete_callback = hour_complete_callback
        self.pick_cache = {}
        self.waveform_cache = {}
        self.associator_cache = {}
        self.associator_cache_lock = Lock()
        self.state_lock = Lock()
        self.completed_intervals = set()
        self.completed_dates = set()
        self.rate_dates = set()
        self._prepare_directories()

    def _prepare_directories(self):
        for path in (
            self.assoc_root / "subnets",
            self.assoc_root / "association_rates",
            self.assoc_root / "hourly_status",
            self.final_root,
            self.final_root / "daily",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for subnet in self.subnet_station_files:
            (self.assoc_root / "subnets" / subnet).mkdir(
                parents=True, exist_ok=True
            )

    def _target_date(self, observed_date):
        return self.target_start <= observed_date < self.target_end

    def _intervals(self, observed_date):
        day_start = utc_day(observed_date) - self.day_shift_seconds
        for index in range(self.intervals_per_day):
            start = day_start + index * self.interval_seconds
            yield start, start + self.interval_seconds

    def _required_dates(self, interval_start, interval_end):
        first = (
            interval_start - self.buffer_seconds + self.day_shift_seconds
        ).date
        last = (
            interval_end + self.buffer_seconds + self.day_shift_seconds
            - 1.0e-6
        ).date
        dates = []
        value = first
        while value <= last:
            dates.append(value)
            value += timedelta(days=1)
        return dates

    def _path_stem(self, interval_start, interval_end):
        return "{}_{}".format(
            _interval_code(interval_start), _interval_code(interval_end)
        )

    def _raw_paths(self, subnet, interval_start, interval_end):
        root = self.assoc_root / "subnets" / subnet
        stem = self._path_stem(interval_start, interval_end)
        return root / ("catalog_" + stem + ".dat"), root / (
            "phase_" + stem + ".dat"
        )

    def _final_paths(self, interval_start, interval_end):
        stem = self._path_stem(interval_start, interval_end)
        return (
            self.final_root / ("catalog_" + stem + ".dat"),
            self.final_root / ("phase_" + stem + ".dat"),
            self.final_root / ("event_groups_" + stem + ".csv"),
        )

    def _associate_subnet(
        self, subnet, interval_start, interval_end, buffered_picks,
    ):
        catalog_path, phase_path = self._raw_paths(
            subnet, interval_start, interval_end
        )
        summary = associate_subnet_picks(
            (interval_start + self.day_shift_seconds).date,
            subnet,
            self.subnet_station_files[subnet],
            buffered_picks,
            catalog_path,
            phase_path,
            self.cfg,
            self.associator_cache,
            self.buffer_seconds,
            associator_cache_lock=self.associator_cache_lock,
        )
        return subnet, phase_path, summary

    def _waveform_context(self, interval_start, interval_end):
        return {
            observed_date: self.waveform_cache[observed_date]
            for observed_date in self._required_dates(
                interval_start, interval_end
            )
            if observed_date in self.waveform_cache
        }

    def _associate_interval(self, interval_start, interval_end):
        key = interval_start.timestamp
        required_dates = self._required_dates(interval_start, interval_end)
        buffered_picks = buffered_pick_interval_arrays(
            [self.pick_cache[value] for value in required_dates],
            interval_start,
            interval_end,
            self.buffer_seconds,
        )

        # NUM_ASSOC_WORKERS is applied across independent hours. Keep subnet
        # work sequential inside each hour to avoid nested thread pools.
        results = [
            self._associate_subnet(
                subnet, interval_start, interval_end, buffered_picks
            )
            for subnet in sorted(self.subnet_station_files)
        ]

        phase_files = {
            subnet: phase_path for subnet, phase_path, _ in results
        }
        final_catalog, final_phase, final_groups = self._final_paths(
            interval_start, interval_end
        )
        if bool(getattr(self.cfg, "enable_post_process", False)):
            # Keep all detections from the buffered association interval.
            # Reassociation may move an origin across the hour boundary, so
            # hourly ownership is assigned only by the postprocess merge.
            merge_summary = merge_buffered_interval_candidates(
                phase_files,
                final_phase,
                final_catalog,
                final_groups,
                self.cfg,
            )
        else:
            merge_summary = merge_canonical_interval(
                interval_start,
                interval_end,
                phase_files,
                final_phase,
                final_catalog,
                final_groups,
                self.cfg,
            )
        summary = {
            "interval_start": str(interval_start),
            "interval_end": str(interval_end),
            "pick_cache_dates": [str(value) for value in required_dates],
            "association_buffer_sec": self.buffer_seconds,
            "subnets": {subnet: item for subnet, _, item in results},
            "merge": merge_summary,
        }
        return {
            "key": key,
            "interval_start": interval_start,
            "interval_end": interval_end,
            "final_catalog": final_catalog,
            "final_phase": final_phase,
            "merge_summary": merge_summary,
            "summary": summary,
        }

    def _complete_interval(self, job):
        key = job["key"]
        interval_start = job["interval_start"]
        interval_end = job["interval_end"]
        final_catalog = job["final_catalog"]
        final_phase = job["final_phase"]
        merge_summary = job["merge_summary"]
        summary = job["summary"]
        with self.state_lock:
            self.completed_intervals.add(key)
        if self.hour_complete_callback is not None:
            postprocess_summary = self.hour_complete_callback(
                interval_start,
                interval_end,
                final_phase,
                final_catalog,
                self._waveform_context(interval_start, interval_end),
                summary,
            )
            if postprocess_summary is not None:
                summary["postprocess"] = postprocess_summary
        status_path = self.assoc_root / "hourly_status" / (
            self._path_stem(interval_start, interval_end) + ".json"
        )
        partial = status_path.with_suffix(status_path.suffix + ".partial")
        partial.write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        partial.replace(status_path)
        final_event_count = merge_summary.get("num_merged_events", 0)
        postprocess = summary.get("postprocess")
        if isinstance(postprocess, dict):
            post_merge = postprocess.get("post_reassociation_merge")
            if isinstance(post_merge, dict):
                final_event_count = post_merge.get(
                    "num_merged_events", final_event_count
                )
        runtime_console.log(
            "hour",
            "{} -- {} | {} final events".format(
                interval_start,
                interval_end,
                final_event_count,
            ),
        )

    def _day_is_complete(self, observed_date):
        return all(
            interval_start.timestamp in self.completed_intervals
            for interval_start, _ in self._intervals(observed_date)
        )

    def _daily_paths(self, observed_date):
        root = self.final_root / "daily"
        code = observed_date.isoformat()
        return root / ("catalog_" + code + ".dat"), root / (
            "phase_" + code + ".dat"
        )

    def _finalize_day(self, observed_date):
        if observed_date in self.completed_dates:
            return
        intervals = list(self._intervals(observed_date))
        daily_catalog, daily_phase = self._daily_paths(observed_date)
        _combine_files(
            [self._final_paths(start, end)[0] for start, end in intervals],
            daily_catalog,
        )
        _combine_files(
            [self._final_paths(start, end)[1] for start, end in intervals],
            daily_phase,
        )

        picks = self.pick_cache.get(observed_date)
        if picks is None:
            raise RuntimeError("daily picks already evicted for {}".format(
                observed_date
            ))
        phase_paths = [daily_phase]
        previous_date = observed_date - timedelta(days=1)
        if self._target_date(previous_date):
            phase_paths.append(self._daily_paths(previous_date)[1])
        rate_path = self.assoc_root / "association_rates" / (
            "association_rate_" + observed_date.isoformat() + ".csv"
        )
        write_association_rate_from_picks(
            observed_date, picks, phase_paths, rate_path,
            interval_start=utc_day(observed_date) - self.day_shift_seconds,
            interval_end=(
                utc_day(observed_date) + 86400 - self.day_shift_seconds
            ),
        )
        self.rate_dates.add(observed_date)
        self.completed_dates.add(observed_date)
        runtime_console.log(
            "day",
            "day={} | {} hourly intervals finalized".format(
                observed_date, len(intervals)
            ),
        )

    def _process_ready_intervals(self, observed_date):
        candidate_dates = [observed_date - timedelta(days=1), observed_date]
        ready = []
        for candidate_date in candidate_dates:
            if not self._target_date(candidate_date):
                continue
            for interval_start, interval_end in self._intervals(candidate_date):
                if interval_start.timestamp in self.completed_intervals:
                    continue
                required_dates = self._required_dates(
                    interval_start, interval_end
                )
                if all(value in self.pick_cache for value in required_dates):
                    ready.append((interval_start, interval_end))

        if ready:
            print(
                "hourly association scheduler: {} intervals | {} workers"
                .format(len(ready), min(self.num_hour_workers, len(ready))),
                flush=True,
            )
            if self.num_hour_workers > 1 and len(ready) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(self.num_hour_workers, len(ready))
                ) as executor:
                    associated = list(executor.map(
                        lambda bounds: self._associate_interval(*bounds), ready
                    ))
            else:
                associated = [
                    self._associate_interval(*bounds) for bounds in ready
                ]
            # One positive-model ensemble is loaded and reused. Complete
            # repicking/reassociation in chronological order after all ready
            # hourly PAL jobs have left the association worker pool.
            for job in associated:
                self._complete_interval(job)

        for candidate_date in candidate_dates:
            if not self._target_date(candidate_date):
                continue
            if self._day_is_complete(candidate_date):
                self._finalize_day(candidate_date)

    def _retain_only_final_hour(self, observed_date):
        day_end = (
            utc_day(observed_date) + 86400 - self.day_shift_seconds
        )
        keep_start = day_end - self.interval_seconds
        keep_end = day_end + self.buffer_seconds
        waveforms = self.waveform_cache.get(observed_date, {})
        waveforms = {
            station: waveform
            for station, waveform in waveforms.items()
            if waveform.trim(keep_start, keep_end)
        }
        self.waveform_cache = {observed_date: waveforms}

    def accept_day(self, date, ensemble_pick_path, summary, waveforms):
        observed_date = date.date
        picks = self.cfg.get_picks(
            utc_day(observed_date), str(Path(ensemble_pick_path).parent)
        )
        self.pick_cache[observed_date] = picks
        self.waveform_cache[observed_date] = waveforms
        print(
            "rolling hourly cache: added {} | {} picks | {} station waveforms".format(
                observed_date, len(picks), len(waveforms)
            ),
            flush=True,
        )
        self._process_ready_intervals(observed_date)
        self._retain_only_final_hour(observed_date)
        self.pick_cache = {
            value: picks
            for value, picks in self.pick_cache.items()
            if value >= observed_date
        }

    def finish(self, output_catalog, output_phase):
        expected_dates = [
            self.target_start + timedelta(days=index)
            for index in range((self.target_end - self.target_start).days)
        ]
        missing_dates = [
            value for value in expected_dates
            if value not in self.completed_dates
        ]
        if missing_dates:
            raise RuntimeError(
                "combined workflow did not finalize dates: {}".format(
                    missing_dates
                )
            )
        _combine_files(
            [self._daily_paths(value)[0] for value in expected_dates],
            output_catalog,
        )
        _combine_files(
            [self._daily_paths(value)[1] for value in expected_dates],
            output_phase,
        )
        self.waveform_cache.clear()


def run_offline_pick_assoc(
    ai_pal_root, cfg, picker_specs, data_dir, full_station_file,
    subnet_station_files, target_time_range, individual_pick_root,
    ensemble_pick_dir, assoc_root, final_root, output_catalog, output_phase,
    num_pick_workers=1, num_assoc_workers=1,
    station_complete_callback=None, day_complete_callback=None,
    hour_complete_callback=None,
    repicker_pos_neg_specs=None, repicker_pos_specs=None,
    overwrite_picks=False,
):
    """Load models once, pick buffered days, and finalize buffered hours."""
    runtime_console.configure(cfg)
    runtime_console.log(
        "AI-PAL",
        "combined local workflow | target={} | workers={}".format(
            target_time_range, max(1, int(num_pick_workers))
        ),
    )
    target_start, target_end = parse_date_range(target_time_range)
    pick_time_range = _range_text(
        target_start - timedelta(days=1), target_end + timedelta(days=1)
    )
    configured_hour_callback = hour_complete_callback
    if bool(getattr(cfg, "enable_post_process", False)):
        if not repicker_pos_neg_specs or not repicker_pos_specs:
            raise ValueError(
                "enable_post_process=True requires both repicker groups"
            )
        cfg.repick_num_workers = max(1, int(num_pick_workers))
        from event_repicker import EventRepicker
        event_repicker = EventRepicker(
            ai_pal_root,
            cfg,
            repicker_pos_neg_specs,
            full_station_file,
            station_dict=load_station_geometry(
                cfg, full_station_file, target_start
            ),
            repicker_pos_specs=repicker_pos_specs,
        )

        def configured_hour_callback(*args):
            (
                interval_start, interval_end, phase_path, catalog_path,
                waveform_context, _summary,
            ) = args
            owner_date = (
                interval_start + float(getattr(cfg, "data_buffer_sec", 0.0))
            ).date
            event_repicker.set_station_geometry(load_station_geometry(
                cfg, full_station_file, owner_date
            ))
            initial_qc = event_repicker.qc_initial_events(
                phase_path, waveform_context, catalog_path
            )
            repick_summary = event_repicker.process_hour(*args)
            repick_summary["initial_waveform_qc"] = initial_qc
            (
                interval_start, interval_end, phase_path, catalog_path,
                _, _,
            ) = args
            phase_path = Path(phase_path)
            groups_path = phase_path.with_name(
                phase_path.name.replace("phase_", "event_groups_", 1)
            ).with_suffix(".csv")
            merge_started = time.perf_counter()
            post_reassociation_merge = merge_canonical_interval(
                interval_start,
                interval_end,
                {"post_reassociation": phase_path},
                phase_path,
                catalog_path,
                groups_path,
                cfg,
            )
            repick_summary["post_reassociation_merge"] = (
                post_reassociation_merge
            )
            repick_summary["post_reassociation_merge_sec"] = (
                time.perf_counter() - merge_started
            )
            final_events = read_phase_file(phase_path)
            waveform_context = args[4]
            if bool(getattr(cfg, "enable_event_waveform_plot", False)):
                repick_summary["num_event_plots"] = (
                    event_repicker.plot_events(
                        final_events,
                        waveform_context,
                        Path(final_root) / "event_waveform",
                    )
                )
            if bool(getattr(
                cfg, "save_filtered_event_waveforms", False
            )):
                repick_summary.update(
                    event_repicker.save_filtered_event_waveforms(
                        final_events,
                        waveform_context,
                        Path(final_root) / "event_waveforms",
                    )
                )
            print(
                "local post-reassociation merge: {} -- {} | {} input "
                "events -> {} events | {} duplicates removed | {:.2f}s"
                .format(
                    interval_start,
                    interval_end,
                    post_reassociation_merge["num_candidate_input_events"],
                    post_reassociation_merge["num_merged_events"],
                    post_reassociation_merge["num_duplicate_events_removed"],
                    repick_summary["post_reassociation_merge_sec"],
                ),
                flush=True,
            )
            if hour_complete_callback is not None:
                extra_summary = hour_complete_callback(*args)
                if extra_summary is not None:
                    repick_summary["extra_callback"] = extra_summary
            return repick_summary

    rolling = RollingHourlyAssociator(
        cfg,
        subnet_station_files,
        assoc_root,
        final_root,
        target_time_range,
        num_workers=num_assoc_workers,
        hour_complete_callback=configured_hour_callback,
    )
    print(
        "combined offline workflow: target {} | daily picking halo {} | "
        "association interval {:.0f}s".format(
            target_time_range, pick_time_range, rolling.interval_seconds
        ),
        flush=True,
    )
    summaries = run_offline_picker_ensemble(
        ai_pal_root=ai_pal_root,
        cfg=cfg,
        picker_specs=picker_specs,
        data_dir=data_dir,
        station_file=full_station_file,
        time_range=pick_time_range,
        individual_pick_root=individual_pick_root,
        ensemble_pick_dir=ensemble_pick_dir,
        num_workers=num_pick_workers,
        station_complete_callback=station_complete_callback,
        day_complete_callback=day_complete_callback,
        day_waveforms_complete_callback=rolling.accept_day,
        pickers_loaded_callback=(
            event_repicker.bind_continuous_pickers
            if bool(getattr(cfg, "enable_post_process", False)) else None
        ),
        overwrite=overwrite_picks,
    )
    rolling.finish(output_catalog, output_phase)
    if bool(getattr(cfg, "enable_post_process", False)):
        event_repicker.close()
    runtime_console.log(
        "output",
        "combined workflow complete | catalog={} | phase={}".format(
            output_catalog, output_phase
        ),
    )
    return summaries
