"""Reusable offline continuous multi-picker inference and ensemble runner."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import ctypes
import gc
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from threading import Lock

import torch
from obspy import UTCDateTime

from pick_ensemble import (
    format_pick_row, merge_picker_pick_files,
)
from picker_stream import PreparedPickerStream, configure_torch_backends
import runtime_console
from rolling_waveform import (
    merge_cached_tail as _merge_cached_tail,
    processing_bounds as _processing_bounds,
    raw_tail as _raw_tail,
)


NATIVE_PICKER_REGISTRY = {
    "SAR": ("picker_SAR", "SAR_Picker"),
    "FT": ("picker_FT", "FT_Picker"),
    "PHN": ("picker_PHN", "PHN_Picker"),
    "RUN": ("picker_RUN", "RUN_Picker"),
}
PICK_OWNERSHIP_VERSION = 1


def _trim_cpu_allocator():
    """Return released station-day arrays to the OS between dates."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _validate_inputs(ai_pal_root, picker_specs, data_dir, station_file):
    if not picker_specs:
        raise ValueError("at least one preferred picker must be selected")
    for required_path in (Path(data_dir), Path(station_file)):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    unknown = {
        spec.get("model", runtime_name)
        for runtime_name, spec in picker_specs.items()
    } - set(NATIVE_PICKER_REGISTRY)
    if unknown:
        raise KeyError("unsupported native pickers: {}".format(sorted(unknown)))
    for runtime_name, spec in picker_specs.items():
        name = spec.get("model", runtime_name)
        if int(spec["gpu_idx"]) < -1:
            raise ValueError("{} gpu_idx must be -1 or >=0".format(name))
        package, _ = NATIVE_PICKER_REGISTRY[name]
        source_dir = Path(ai_pal_root) / package
        for required_path in (
            source_dir, source_dir / "picker.py", source_dir / "models.py",
            Path(spec["config"]),
        ):
            if not required_path.exists():
                raise FileNotFoundError(required_path)
        if not spec.get("ckpt"):
            raise ValueError("{} requires an explicit ckpt file path, not ckpt_dir".format(runtime_name))
        checkpoint = Path(spec["ckpt"])
        if checkpoint.is_file():
            continue
        raise FileNotFoundError(
            "{} checkpoint must be an existing file: {}"
            .format(runtime_name, checkpoint)
        )


def _load_runtime_module(ai_pal_root, package, runtime_name, config_path):
    """Load one picker with a private case config under a unique module name."""
    package_module = importlib.import_module(package)
    config_key = package + ".config"
    models_key = package + ".models"
    old_config_module = sys.modules.get(config_key)
    old_models_module = sys.modules.get(models_key)
    old_config_attribute = getattr(package_module, "config", None)
    old_models_attribute = getattr(package_module, "models", None)
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in runtime_name
    )
    config_module_name = "{}_offline_config_{}".format(package, safe_name)
    config_spec = importlib.util.spec_from_file_location(
        config_module_name, str(config_path)
    )
    if config_spec is None or config_spec.loader is None:
        raise ImportError("cannot load picker config from {}".format(config_path))
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    picker_path = Path(ai_pal_root) / package / "picker.py"
    models_path = Path(ai_pal_root) / package / "models.py"
    module_name = "{}.picker_offline_{}".format(package, safe_name)
    picker_spec = importlib.util.spec_from_file_location(
        module_name, str(picker_path)
    )
    if picker_spec is None or picker_spec.loader is None:
        raise ImportError("cannot load picker module from {}".format(picker_path))
    try:
        sys.modules[config_key] = config_module
        setattr(package_module, "config", config_module)
        models_module_name = "{}.models_offline_{}".format(
            package, safe_name
        )
        models_spec = importlib.util.spec_from_file_location(
            models_module_name, str(models_path)
        )
        if models_spec is None or models_spec.loader is None:
            raise ImportError(
                "cannot load picker models from {}".format(models_path)
            )
        models_module = importlib.util.module_from_spec(models_spec)
        sys.modules[models_module_name] = models_module
        models_spec.loader.exec_module(models_module)
        sys.modules[models_key] = models_module
        setattr(package_module, "models", models_module)
        module = importlib.util.module_from_spec(picker_spec)
        sys.modules[module_name] = module
        picker_spec.loader.exec_module(module)
        return module
    finally:
        if old_config_module is None:
            sys.modules.pop(config_key, None)
        else:
            sys.modules[config_key] = old_config_module
        if old_config_attribute is None:
            try:
                delattr(package_module, "config")
            except AttributeError:
                pass
        else:
            setattr(package_module, "config", old_config_attribute)
        if old_models_module is None:
            sys.modules.pop(models_key, None)
        else:
            sys.modules[models_key] = old_models_module
        if old_models_attribute is None:
            try:
                delattr(package_module, "models")
            except AttributeError:
                pass
        else:
            setattr(package_module, "models", old_models_attribute)


def _load_pickers(ai_pal_root, picker_specs):
    pickers = {}
    for runtime_name, spec in picker_specs.items():
        model_name = spec.get("model", runtime_name)
        package, class_name = NATIVE_PICKER_REGISTRY[model_name]
        module = _load_runtime_module(
            ai_pal_root, package, runtime_name, spec["config"]
        )
        picker_class = getattr(module, class_name)
        device = "CPU" if int(spec["gpu_idx"]) == -1 else "GPU {}".format(
            spec["gpu_idx"]
        )
        print("loading {} on {} from {}".format(
            runtime_name, device,
            spec["ckpt"],
        ), flush=True)
        pickers[runtime_name] = picker_class(
            str(spec["ckpt"]),
            -1,
            spec["gpu_idx"],
        )
    return pickers


def _group_by_device(pickers):
    groups = {}
    for name, picker in pickers.items():
        groups.setdefault(str(picker.device), []).append((name, picker))
    print("picker device groups: {}".format({
        device: [name for name, _ in members]
        for device, members in groups.items()
    }), flush=True)
    runtime_console.log(
        "startup",
        "picker devices={}".format({
            device: [name for name, _ in members]
            for device, members in groups.items()
        }),
    )
    return groups


def _run_device_group(
    members, prepared, device_lock, start_time, end_time,
    defer_waveform_qc=False,
):
    results = {}
    with device_lock:
        for name, picker in members:
            print("picking {} with {}".format(prepared.net_sta, name))
            with torch.inference_mode():
                picks = picker.pick(
                    prepared.stream,
                    pick_start_time=start_time,
                    pick_end_time=end_time,
                    prepared=prepared,
                    defer_waveform_qc=defer_waveform_qc,
                )
            results[name] = picks if picks else []
    return results


def _read_current_station_stream(
    net_sta, data_paths, cfg, stations, start_time, end_time,
):
    normalize_to_three_channels = getattr(
        cfg, "normalize_to_three_channels", True
    )
    return cfg.read_data(
        data_paths,
        stations,
        start_time=start_time,
        end_time=end_time,
        normalize_to_three_channels=normalize_to_three_channels,
        to_prep=bool(getattr(cfg, "to_prep", True)),
        location_priority=getattr(
            cfg, "location_priority", ("10", "20", "01", "02", "00", "")
        ),
        channel_priority=getattr(
            cfg, "channel_priority", ("HH", "BH", "EH", "HN", "EN", "SH")
        ),
    )


def _prepare_and_pick_station(
    net_sta, data_paths, cfg, pickers, device_groups, device_locks,
    stations, day_start, day_end, buffer_sec, station_complete_callback=None,
    retain_waveform=False, previous_raw_tail=None,
):
    print("-" * 60)
    print("reading and preprocessing {} {} once".format(
        net_sta, day_start.date
    ))
    current = _read_current_station_stream(
        net_sta, data_paths, cfg, stations, day_start, day_end
    )
    next_raw_tail = _raw_tail(current, day_end, buffer_sec)
    stream = _merge_cached_tail(
        current,
        previous_raw_tail,
        day_start - 2.0 * buffer_sec,
        day_end,
    )
    current = None
    prepared = PreparedPickerStream.from_raw_stream(stream, cfg)
    # Preprocessing has transferred the usable arrays into `prepared`.
    # Do not keep the original read stream alive during model inference.
    stream = None
    if prepared is None:
        print("skip {}: unusable waveform after preprocessing".format(net_sta))
        return net_sta, {}, None, next_raw_tail

    try:
        station_results = {}
        pick_start, pick_end = _processing_bounds(day_start, buffer_sec)
        if device_groups:
            with ThreadPoolExecutor(max_workers=len(device_groups)) as executor:
                futures = [
                    executor.submit(
                        _run_device_group,
                        members,
                        prepared,
                        device_locks[device],
                        pick_start,
                        pick_end,
                        retain_waveform,
                    )
                    for device, members in device_groups.items()
                ]
                for future in futures:
                    station_results.update(future.result())
        if station_complete_callback is not None:
            station_complete_callback(
                day_start, net_sta, prepared, station_results
            )
        retained = prepared.retained_waveform() if retain_waveform else None
        return net_sta, station_results, retained, next_raw_tail
    finally:
        # Pick-only runs release the station-day stream and every device copy
        # here. Combined runs transfer the filtered stream to `retained` first.
        prepared.release()


def _pick_one_day(
    date, cfg, pickers, device_groups, device_locks, data_dir, stations,
    pick_dirs, num_workers, station_complete_callback=None,
    retain_waveforms=False, previous_raw_tails=None,
):
    buffer_sec = float(cfg.data_buffer_sec)
    day_start, day_end = date, date + 86400
    normalize_to_three_channels = getattr(cfg, "normalize_to_three_channels", True)
    data_dict = cfg.get_data_dict(
        date,
        str(data_dir),
        normalize_to_three_channels=normalize_to_three_channels,
    )
    previous_raw_tails = previous_raw_tails or {}
    pick_paths = {
        name: pick_dirs[name] / "{}.pick".format(date.date)
        for name in pickers
    }
    partial_paths = {
        name: path.with_suffix(path.suffix + ".partial")
        for name, path in pick_paths.items()
    }
    outputs = {
        name: partial_paths[name].open("w", encoding="utf-8")
        for name in pickers
    }
    try:
        station_items = [
            (net_sta, data_paths)
            for net_sta, data_paths in sorted(data_dict.items())
            if net_sta in stations
        ]

        def process(item):
            return _prepare_and_pick_station(
                item[0], item[1], cfg, pickers, device_groups, device_locks,
                stations, day_start, day_end, buffer_sec,
                station_complete_callback,
                retain_waveforms,
                previous_raw_tails.get(item[0]),
            )

        if num_workers > 1:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            result_iter = executor.map(process, station_items)
        else:
            executor = None
            result_iter = map(process, station_items)

        try:
            retained_waveforms = {}
            next_raw_tails = {}
            for net_sta, station_results, retained, next_raw_tail in result_iter:
                for name, picks in station_results.items():
                    for pick in picks:
                        outputs[name].write(format_pick_row(pick))
                if retained is not None:
                    retained_waveforms[net_sta] = retained
                if next_raw_tail:
                    next_raw_tails[net_sta] = next_raw_tail
        finally:
            if executor is not None:
                executor.shutdown()
    except BaseException:
        for output in outputs.values():
            output.close()
        for path in partial_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for output in outputs.values():
            output.close()
        for name, path in pick_paths.items():
            partial_paths[name].replace(path)
    return (
        pick_paths,
        retained_waveforms if retain_waveforms else None,
        next_raw_tails,
    )


def _date_list(time_range):
    start, end = [UTCDateTime(value) for value in time_range.split("-")]
    return [
        start + offset * 86400
        for offset in range(int((end - start) / 86400))
    ]


def _load_station_selectors(cfg, station_file, dates):
    """Load static station metadata or the union of date-aware epochs."""
    get_sta_dict = cfg.get_sta_dict
    if len(inspect.signature(get_sta_dict).parameters) < 2:
        return get_sta_dict(str(station_file))
    stations = {}
    for date in dates:
        stations.update(get_sta_dict(str(station_file), date))
    return stations


def _prepare_day_waveforms(
    date, cfg, data_dir, stations, num_workers, previous_raw_tails,
):
    """Rebuild filtered context for downstream repicking without inference."""
    buffer_sec = float(cfg.data_buffer_sec)
    day_start, day_end = date, date + 86400
    data_dict = cfg.get_data_dict(
        date,
        str(data_dir),
        normalize_to_three_channels=bool(getattr(
            cfg, "normalize_to_three_channels", True
        )),
    )
    station_items = [
        (net_sta, data_paths)
        for net_sta, data_paths in sorted(data_dict.items())
        if net_sta in stations
    ]

    def process(item):
        return _prepare_and_pick_station(
            item[0], item[1], cfg, {}, {}, {}, stations,
            day_start, day_end, buffer_sec, retain_waveform=True,
            previous_raw_tail=previous_raw_tails.get(item[0]),
        )

    if num_workers > 1 and len(station_items) > 1:
        with ThreadPoolExecutor(
            max_workers=min(num_workers, len(station_items))
        ) as executor:
            results = executor.map(process, station_items)
            retained = {}
            next_raw_tails = {}
            for net_sta, _, waveform, next_raw_tail in results:
                if waveform is not None:
                    retained[net_sta] = waveform
                if next_raw_tail:
                    next_raw_tails[net_sta] = next_raw_tail
    else:
        retained = {}
        next_raw_tails = {}
        for item in station_items:
            net_sta, _, waveform, next_raw_tail = process(item)
            if waveform is not None:
                retained[net_sta] = waveform
            if next_raw_tail:
                next_raw_tails[net_sta] = next_raw_tail
    return retained, next_raw_tails


def _load_raw_tail_cache(date, cfg, data_dir, stations, num_workers):
    """Load only one day's final raw tail, used to seed or advance a resumed run."""
    buffer_sec = float(cfg.data_buffer_sec)
    if buffer_sec <= 0:
        return {}
    day_end = date + 86400
    data_dict = cfg.get_data_dict(
        date,
        str(data_dir),
        normalize_to_three_channels=bool(getattr(
            cfg, "normalize_to_three_channels", True
        )),
    )
    station_items = [
        (net_sta, data_paths)
        for net_sta, data_paths in sorted(data_dict.items())
        if net_sta in stations
    ]

    def process(item):
        stream = _read_current_station_stream(
            item[0], item[1], cfg, stations,
            day_end - 2.0 * buffer_sec, day_end,
        )
        return item[0], _raw_tail(stream, day_end, buffer_sec)

    if num_workers > 1 and len(station_items) > 1:
        with ThreadPoolExecutor(
            max_workers=min(num_workers, len(station_items))
        ) as executor:
            results = executor.map(process, station_items)
            return {
                net_sta: tail for net_sta, tail in results if tail
            }
    return {
        net_sta: tail
        for net_sta, tail in map(process, station_items)
        if tail
    }


def _existing_pick_summary(path):
    with Path(path).open(encoding="utf-8") as fp:
        count = sum(1 for line in fp if line.strip())
    return {
        "output_path": str(path),
        "num_merged_picks": count,
        "input_counts": {"existing_ensemble": count},
        "skipped_existing": True,
    }


def _ownership_path(ensemble_path):
    return Path(str(ensemble_path) + ".ownership.json")


def _has_current_ownership(ensemble_path, buffer_sec):
    path = _ownership_path(ensemble_path)
    if not path.exists():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return (
            int(metadata.get("version", -1)) == PICK_OWNERSHIP_VERSION
            and float(metadata.get("data_buffer_sec")) == float(buffer_sec)
            and metadata.get("interval")
            == "[D-data_buffer_sec,D+1day-data_buffer_sec)"
        )
    except (OSError, TypeError, ValueError):
        return False


def _write_ownership(ensemble_path, buffer_sec):
    path = _ownership_path(ensemble_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps({
        "version": PICK_OWNERSHIP_VERSION,
        "data_buffer_sec": float(buffer_sec),
        "interval": "[D-data_buffer_sec,D+1day-data_buffer_sec)",
    }, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def run_offline_picker_ensemble(
    ai_pal_root, cfg, picker_specs, data_dir, station_file, time_range,
    individual_pick_root, ensemble_pick_dir, num_workers=1,
    day_complete_callback=None, station_complete_callback=None,
    day_waveforms_complete_callback=None,
    pickers_loaded_callback=None,
    overwrite=False,
):
    """Run configured native pickers and write individual/ensemble daily picks."""
    runtime_console.configure(cfg)
    configure_torch_backends(cfg)
    _validate_inputs(
        ai_pal_root, picker_specs, data_dir, station_file
    )
    if any(spec.get("group", "POS_NEG") != "POS_NEG" for spec in picker_specs.values()):
        raise ValueError("continuous picking only supports POS_NEG models")
    required = int(cfg.picker_pos_neg_group_min_picker_support)
    if required <= 0 or required > len(picker_specs):
        raise ValueError("POS_NEG continuous picker support {} is invalid for {} models".format(
            required, len(picker_specs)))
    dates = _date_list(time_range)
    runtime_console.log(
        "AI-PAL",
        "local picking | days={} | models={} | workers={}".format(
            len(dates), ",".join(picker_specs), max(1, int(num_workers))
        ),
    )
    pickers = _load_pickers(ai_pal_root, picker_specs)
    if pickers_loaded_callback is not None:
        pickers_loaded_callback({"POS_NEG": {
            picker_specs[name].get("model", name): picker
            for name, picker in pickers.items()
        }})
    device_groups = _group_by_device(pickers)
    num_workers = max(1, int(num_workers))
    device_locks = {device: Lock() for device in device_groups}
    print(
        "offline station-date workers: {} | one inference queue per device".format(
            num_workers
        ),
        flush=True,
    )
    stations = _load_station_selectors(cfg, station_file, dates)
    print("full station file: {} | {} selectors".format(
        station_file, len(stations)
    ))
    buffer_sec = float(cfg.data_buffer_sec)
    if buffer_sec < 0:
        raise ValueError("data_buffer_sec must be nonnegative")
    taper_sec = float(getattr(cfg, "taper_max_length_sec", 0.0))
    if buffer_sec and buffer_sec < taper_sec:
        raise ValueError(
            "data_buffer_sec must be at least taper_max_length_sec"
        )
    if dates and buffer_sec:
        previous_raw_tails = _load_raw_tail_cache(
            dates[0] - 86400, cfg, data_dir, stations, num_workers
        )
    else:
        previous_raw_tails = {}

    if cfg.save_individual_picker_outputs:
        context = nullcontext(Path(individual_pick_root))
    else:
        context = tempfile.TemporaryDirectory(prefix="ai-pal-picker-branches-")

    with context as individual_root:
        pick_dirs = {}
        for index, name in enumerate(pickers, start=1):
            spec = picker_specs[name]
            model_name = spec.get("model", name)
            indexed_name = "1.1.{}_picks_pos_neg_{}".format(
                index, model_name
            )
            indexed_path = Path(individual_root) / indexed_name
            legacy_path = Path(individual_root) / "picks_{}".format(name)
            if (
                cfg.save_individual_picker_outputs
                and legacy_path.is_dir()
                and not indexed_path.exists()
            ):
                legacy_path.replace(indexed_path)
                print(
                    "migrated legacy local picker directory: {} -> {}"
                    .format(legacy_path, indexed_path),
                    flush=True,
                )
            pick_dirs[name] = indexed_path
        for path in pick_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        summaries = []
        need_retained_waveforms = (
            day_waveforms_complete_callback is not None
            and bool(getattr(cfg, "enable_post_process", False))
        )
        for index, date in enumerate(dates, start=1):
            filename = "{}.pick".format(date.date)
            ensemble_path = Path(ensemble_pick_dir) / filename
            runtime_console.log(
                "day",
                "{} picking start | {}/{}".format(
                    date.date, index, len(dates)
                ),
            )
            individual_complete = all(
                (directory / filename).exists()
                for directory in pick_dirs.values()
            )
            complete = not overwrite and ensemble_path.exists() and (
                not cfg.save_individual_picker_outputs
                or individual_complete
            ) and _has_current_ownership(ensemble_path, buffer_sec)
            retained_waveforms = {}
            next_raw_tails = {}
            if complete:
                day_summaries = [
                    _existing_pick_summary(ensemble_path)
                ]
                if need_retained_waveforms:
                    print(
                        "existing picks found for {}; rebuilding filtered "
                        "waveform context without picker inference".format(
                            date.date
                        ),
                        flush=True,
                    )
                    retained_waveforms, next_raw_tails = _prepare_day_waveforms(
                        date, cfg, data_dir, stations, num_workers,
                        previous_raw_tails,
                    )
                else:
                    next_raw_tails = _load_raw_tail_cache(
                        date, cfg, data_dir, stations, num_workers
                    )
                print(
                    "skip completed daily picking: {}".format(ensemble_path),
                    flush=True,
                )
            elif (
                not overwrite
                and individual_complete
                and _has_current_ownership(ensemble_path, buffer_sec)
            ):
                day_summaries = [merge_picker_pick_files(
                    {name: path / filename for name, path in pick_dirs.items()},
                    ensemble_path, cfg.tp_dev, cfg.ts_dev, min_support=required,
                )]
                _write_ownership(ensemble_path, buffer_sec)
                if need_retained_waveforms:
                    retained_waveforms, next_raw_tails = _prepare_day_waveforms(
                        date, cfg, data_dir, stations, num_workers,
                        previous_raw_tails,
                    )
                else:
                    next_raw_tails = _load_raw_tail_cache(
                        date, cfg, data_dir, stations, num_workers
                    )
                print(
                    "rebuilt missing ensemble from completed picker files: {}"
                    .format(ensemble_path),
                    flush=True,
                )
            else:
                _ownership_path(ensemble_path).unlink(missing_ok=True)
                paths, retained_waveforms, next_raw_tails = _pick_one_day(
                    date, cfg, pickers, device_groups, device_locks, data_dir,
                    stations, pick_dirs, num_workers,
                    station_complete_callback,
                    retain_waveforms=need_retained_waveforms,
                    previous_raw_tails=previous_raw_tails,
                )
                # Mark the finalized individual files before ensemble merging,
                # so a merge-only retry can distinguish them from legacy picks.
                _write_ownership(ensemble_path, buffer_sec)
                day_summaries = [merge_picker_pick_files(
                    {name: path / filename for name, path in pick_dirs.items()},
                    ensemble_path, cfg.tp_dev, cfg.ts_dev, min_support=required,
                )]
                _write_ownership(ensemble_path, buffer_sec)
            summaries.extend(day_summaries)
            runtime_console.log(
                "day",
                "day={} | picks={} | progress={}/{}".format(
                    date.date,
                    int(day_summaries[0].get("num_merged_picks", 0)),
                    index,
                    len(dates),
                ),
            )
            print("{} / {} days complete: {}".format(
                index, len(dates), ensemble_path
            ), flush=True)
            if day_complete_callback is not None:
                day_complete_callback(
                    date,
                    Path(ensemble_pick_dir) / filename,
                    day_summaries[0],
                )
            if day_waveforms_complete_callback is not None:
                day_waveforms_complete_callback(
                    date,
                    Path(ensemble_pick_dir) / filename,
                    day_summaries[0],
                    retained_waveforms,
                )
            previous_raw_tails = next_raw_tails
            _trim_cpu_allocator()
    num_input = sum(sum(item["input_counts"].values()) for item in summaries)
    num_output = sum(item["num_merged_picks"] for item in summaries)
    print("picker ensemble: {} input picks -> {} consensus picks in {} files".format(
        num_input, num_output, len(summaries)
    ), flush=True)
    print("ensemble pick directory: {}".format(ensemble_pick_dir), flush=True)
    runtime_console.log(
        "output",
        "picking complete | days={} | consensus_picks={} | {}".format(
            len(summaries), num_output, ensemble_pick_dir
        ),
    )
    if not cfg.save_individual_picker_outputs:
        print("temporary individual picker outputs removed", flush=True)
    return summaries
