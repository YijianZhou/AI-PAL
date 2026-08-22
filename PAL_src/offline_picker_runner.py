"""Reusable offline continuous multi-picker inference and ensemble runner."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import ctypes
import gc
import importlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Lock

import torch
from obspy import UTCDateTime

from pick_ensemble import (
    format_pick_row, merge_picker_pick_files, read_pick_file,
)
from picker_stream import PreparedPickerStream, configure_torch_backends


NATIVE_PICKER_REGISTRY = {
    "SAR": ("picker_SAR", "SAR_Picker"),
    "FT": ("picker_FT", "FT_Picker"),
    "PHN": ("picker_PHN", "PHN_Picker"),
    "RUN": ("picker_RUN", "RUN_Picker"),
}


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
        checkpoint = Path(spec.get("ckpt", spec.get("ckpt_dir", "")))
        if checkpoint.is_dir() and any(checkpoint.glob("*.ckpt")):
            continue
        if checkpoint.is_file():
            continue
        raise FileNotFoundError(
            "{} checkpoint file/directory was not found or is empty: {}"
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
            spec.get("ckpt", spec.get("ckpt_dir")),
        ), flush=True)
        pickers[runtime_name] = picker_class(
            str(spec.get("ckpt", spec.get("ckpt_dir"))),
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


def _prepare_and_pick_station(
    net_sta, data_paths, cfg, pickers, device_groups, device_locks,
    stations, day_start, day_end, buffer_sec, station_complete_callback=None,
    retain_waveform=False,
):
    print("-" * 60)
    print("reading and preprocessing {} {} once".format(
        net_sta, day_start.date
    ))
    normalize_to_three_channels = getattr(
        cfg, "normalize_to_three_channels", True
    )
    stream = cfg.read_data(
        data_paths,
        stations,
        start_time=day_start - buffer_sec,
        end_time=day_end + buffer_sec,
        normalize_to_three_channels=normalize_to_three_channels,
    )
    prepared = PreparedPickerStream.from_raw_stream(stream, cfg)
    # Preprocessing has transferred the usable arrays into `prepared`.
    # Do not keep the original read stream alive during model inference.
    stream = None
    if prepared is None:
        print("skip {}: unusable waveform after preprocessing".format(net_sta))
        return net_sta, {}, None

    try:
        station_results = {}
        if device_groups:
            with ThreadPoolExecutor(max_workers=len(device_groups)) as executor:
                futures = [
                    executor.submit(
                        _run_device_group,
                        members,
                        prepared,
                        device_locks[device],
                        day_start,
                        day_end,
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
        return net_sta, station_results, retained
    finally:
        # Pick-only runs release the station-day stream and every device copy
        # here. Combined runs transfer the filtered stream to `retained` first.
        prepared.release()


def _pick_one_day(
    date, cfg, pickers, device_groups, device_locks, data_dir, stations,
    pick_dirs, num_workers, station_complete_callback=None,
    retain_waveforms=False,
):
    buffer_sec = float(cfg.data_buffer_sec)
    day_start, day_end = date, date + 86400
    normalize_to_three_channels = getattr(cfg, "normalize_to_three_channels", True)
    data_dict = cfg.get_buffered_data_dict(
        date,
        str(data_dir),
        buffer_sec,
        normalize_to_three_channels=normalize_to_three_channels,
    )
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
            )

        if num_workers > 1:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            result_iter = executor.map(process, station_items)
        else:
            executor = None
            result_iter = map(process, station_items)

        try:
            retained_waveforms = {}
            for net_sta, station_results, retained in result_iter:
                for name, picks in station_results.items():
                    for pick in picks:
                        outputs[name].write(format_pick_row(pick))
                if retained is not None:
                    retained_waveforms[net_sta] = retained
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
    return pick_paths, retained_waveforms if retain_waveforms else None


def _date_list(time_range):
    start, end = [UTCDateTime(value) for value in time_range.split("-")]
    return [
        start + offset * 86400
        for offset in range(int((end - start) / 86400))
    ]


def _prepare_day_waveforms(
    date, cfg, data_dir, stations, num_workers,
):
    """Rebuild filtered context for downstream repicking without inference."""
    buffer_sec = float(cfg.data_buffer_sec)
    day_start, day_end = date, date + 86400
    data_dict = cfg.get_buffered_data_dict(
        date,
        str(data_dir),
        buffer_sec,
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
        )

    if num_workers > 1 and len(station_items) > 1:
        with ThreadPoolExecutor(
            max_workers=min(num_workers, len(station_items))
        ) as executor:
            results = executor.map(process, station_items)
            retained = {
                net_sta: waveform
                for net_sta, _, waveform in results
                if waveform is not None
            }
    else:
        retained = {}
        for item in station_items:
            net_sta, _, waveform = process(item)
            if waveform is not None:
                retained[net_sta] = waveform
    return retained


def _existing_pick_summary(path):
    with Path(path).open(encoding="utf-8") as fp:
        count = sum(1 for line in fp if line.strip())
    return {
        "output_path": str(path),
        "num_merged_picks": count,
        "input_counts": {"existing_ensemble": count},
        "skipped_existing": True,
    }


def _merge_grouped_picker_file(
    pick_dirs, picker_groups, group_support, group_root, ensemble_path,
    filename, tp_dev, ts_dev,
):
    """Apply per-group consensus, then merge accepted groups equally."""
    group_files = {}
    for group_name, members in picker_groups.items():
        group_path = Path(group_root) / group_name / filename
        merge_picker_pick_files(
            {name: pick_dirs[name] / filename for name in members},
            group_path,
            tp_dev,
            ts_dev,
            min_support=group_support[group_name],
        )
        group_files[group_name] = group_path
    summary = merge_picker_pick_files(
        group_files,
        ensemble_path,
        tp_dev,
        ts_dev,
        min_support=1,
    )
    # Group names are voting identities, while model names are the useful
    # provenance exposed to association and phase products.
    records = read_pick_file(ensemble_path)
    partial = Path(str(ensemble_path) + ".partial")
    with partial.open("w", encoding="utf-8") as fp:
        for record in records:
            record["sources"] = sorted(record["picker_cluster_sizes"])
            record["num_support"] = len(record["sources"])
            fp.write(format_pick_row(record))
    partial.replace(ensemble_path)
    return summary


def run_offline_picker_ensemble(
    ai_pal_root, cfg, picker_specs, data_dir, station_file, time_range,
    individual_pick_root, ensemble_pick_dir, num_workers=1,
    day_complete_callback=None, station_complete_callback=None,
    day_waveforms_complete_callback=None,
    pickers_loaded_callback=None,
    overwrite=False,
):
    """Run configured native pickers and write individual/ensemble daily picks."""
    configure_torch_backends(cfg)
    _validate_inputs(
        ai_pal_root, picker_specs, data_dir, station_file
    )
    picker_groups = {}
    for runtime_name, spec in picker_specs.items():
        picker_groups.setdefault(spec.get("group", "POS_NEG"), []).append(
            runtime_name
        )
    group_support = {
        "POS_NEG": int(cfg.picker_pos_neg_group_min_picker_support),
        "POS": int(cfg.picker_pos_group_min_picker_support),
    }
    for group_name, members in picker_groups.items():
        required = group_support[group_name]
        if required <= 0 or required > len(members):
            raise ValueError(
                "{} continuous picker support {} is invalid for {} models"
                .format(group_name, required, len(members))
            )
    dates = _date_list(time_range)
    pickers = _load_pickers(ai_pal_root, picker_specs)
    if pickers_loaded_callback is not None:
        grouped_pickers = {}
        for runtime_name, picker in pickers.items():
            spec = picker_specs[runtime_name]
            grouped_pickers.setdefault(
                spec.get("group", "POS_NEG"), {}
            )[spec.get("model", runtime_name)] = picker
        pickers_loaded_callback(grouped_pickers)
    device_groups = _group_by_device(pickers)
    num_workers = max(1, int(num_workers))
    device_locks = {device: Lock() for device in device_groups}
    print(
        "offline station-date workers: {} | one inference queue per device".format(
            num_workers
        ),
        flush=True,
    )
    stations = cfg.get_sta_dict(str(station_file))
    print("full station file: {} | {} selectors".format(
        station_file, len(stations)
    ))

    if cfg.save_individual_picker_outputs:
        context = nullcontext(Path(individual_pick_root))
    else:
        context = tempfile.TemporaryDirectory(prefix="ai-pal-picker-branches-")

    with context as individual_root:
        pick_dirs = {}
        for index, name in enumerate(pickers, start=1):
            spec = picker_specs[name]
            group_name = spec.get("group", "POS_NEG").lower()
            model_name = spec.get("model", name)
            indexed_name = "1.1.{}_picks_{}_{}".format(
                index, group_name, model_name
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
        group_root = Path(individual_root) / ".picker_group_consensus"
        summaries = []
        need_retained_waveforms = (
            day_waveforms_complete_callback is not None
            and bool(getattr(cfg, "enable_post_process", False))
        )
        for index, date in enumerate(dates, start=1):
            filename = "{}.pick".format(date.date)
            ensemble_path = Path(ensemble_pick_dir) / filename
            individual_complete = all(
                (directory / filename).exists()
                for directory in pick_dirs.values()
            )
            complete = not overwrite and ensemble_path.exists() and (
                not cfg.save_individual_picker_outputs
                or individual_complete
            )
            retained_waveforms = {}
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
                    retained_waveforms = _prepare_day_waveforms(
                        date, cfg, data_dir, stations, num_workers
                    )
                print(
                    "skip completed daily picking: {}".format(ensemble_path),
                    flush=True,
                )
            elif not overwrite and individual_complete:
                day_summaries = [_merge_grouped_picker_file(
                    pick_dirs, picker_groups, group_support, group_root,
                    ensemble_path, filename, cfg.tp_dev, cfg.ts_dev,
                )]
                if need_retained_waveforms:
                    retained_waveforms = _prepare_day_waveforms(
                        date, cfg, data_dir, stations, num_workers
                    )
                print(
                    "rebuilt missing ensemble from completed picker files: {}"
                    .format(ensemble_path),
                    flush=True,
                )
            else:
                paths, retained_waveforms = _pick_one_day(
                    date, cfg, pickers, device_groups, device_locks, data_dir,
                    stations, pick_dirs, num_workers,
                    station_complete_callback,
                    retain_waveforms=need_retained_waveforms,
                )
                day_summaries = [_merge_grouped_picker_file(
                    pick_dirs, picker_groups, group_support, group_root,
                    ensemble_path, filename, cfg.tp_dev, cfg.ts_dev,
                )]
            summaries.extend(day_summaries)
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
            _trim_cpu_allocator()
        shutil.rmtree(group_root, ignore_errors=True)
    num_input = sum(sum(item["input_counts"].values()) for item in summaries)
    num_output = sum(item["num_merged_picks"] for item in summaries)
    print("picker ensemble: {} input picks -> {} consensus picks in {} files".format(
        num_input, num_output, len(summaries)
    ), flush=True)
    print("ensemble pick directory: {}".format(ensemble_pick_dir), flush=True)
    if not cfg.save_individual_picker_outputs:
        print("temporary individual picker outputs removed", flush=True)
    return summaries
