"""Run realtime picking, preferred PAL association, and explicit references.

This script is intended to be launched from
``3_run_ai_pal/run_ai_pal_realtime/1_run_ai_pal_pick_assoc_realtime_eg.py`` after staging
the shared AI-PAL config in ``PAL_src`` and each enabled native model config
in its picker package.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import glob
import importlib
import json
import importlib.util
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAL_DIR = BASE_DIR
AI_PAL_ROOT = os.path.dirname(BASE_DIR)
if AI_PAL_ROOT not in sys.path:
    sys.path.insert(0, AI_PAL_ROOT)

import config_ai_pal
import realtime_pipeline as rtp
import realtime_pickers
import runtime_console
from reference_association import reference_workflows, run_gamma, require_gamma_runtime
from picker_stream import configure_torch_backends
from station_sets import build_station_union
# Load PAL under a unique module name without changing global import precedence.
_ASSOCIATOR_PATH = os.path.join(PAL_DIR, "associator_pal.py")
_assoc_spec = importlib.util.spec_from_file_location(
    "ai_pal_associator_pal", _ASSOCIATOR_PATH
)
if _assoc_spec is None or _assoc_spec.loader is None:
    raise ImportError("cannot load PAL associator from {}".format(_ASSOCIATOR_PATH))
associator_pal = importlib.util.module_from_spec(_assoc_spec)
_assoc_spec.loader.exec_module(associator_pal)


def _load_picker_config(path, module_name):
    if not path:
        raise ValueError("missing config path for {}".format(module_name))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load picker config from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config()

def parse_args():
    cfg = config_ai_pal.Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_sta_file", type=str, default=None)
    parser.add_argument("--subnet_sta_files", nargs="*", default=[])
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--in_glob", type=str, default="*.ms")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--native_pickers_json", type=str, required=True)
    parser.add_argument("--reference_pickers_json", type=str, required=True)
    parser.add_argument("--reference_associators_json", type=str, default="{}")
    parser.add_argument("--repicker_pos_neg_json", type=str, default="{}")
    parser.add_argument("--repicker_pos_json", type=str, default="{}")
    return cfg, parser.parse_args()


def _enabled_picker_names(cfg):
    names = list(cfg.continuous_picker_specs)
    names.extend(
        name for name in cfg.reference_picker_names if name not in names
    )
    supported = {"SAR", "FT", "PHN", "RUN", "PHN-SB"}
    unknown = [
        spec["model"] for spec in cfg.continuous_picker_specs.values()
        if spec["model"] not in supported
    ]
    unknown.extend(
        name for name in cfg.reference_picker_names if name not in {"PHN-SB"}
    )
    if unknown:
        raise ValueError("unsupported realtime pickers: {}".format(unknown))
    return names


def _association_branch_members(cfg):
    preferred = [
        name
        for members in cfg.continuous_picker_groups.values()
        for name in members
    ]
    branches = {rtp.PREFERRED_ENSEMBLE_BRANCH: preferred}
    for name, workflow in reference_workflows(cfg).items():
        branches[name] = [workflow["picker"]]
    return branches


def apply_overrides(cfg, args):
    cfg.out_root = args.out_root
    if args.full_sta_file is None:
        cfg.full_sta_file = str(build_station_union(
            args.subnet_sta_files,
            os.path.join(
                cfg.out_root, "_internal", "stations", "picking_union.sta"
            ),
        ))
        print(
            "FULL_STATION_FILE=None: picking station union written to {}"
            .format(cfg.full_sta_file),
            flush=True,
        )
    else:
        cfg.full_sta_file = args.full_sta_file
    if args.subnet_sta_files:
        cfg.association_station_files = {
            subnet_name_from_path(path): path for path in args.subnet_sta_files
        }
        if len(cfg.association_station_files) != len(args.subnet_sta_files):
            raise ValueError("subnet station filenames must resolve to unique names")
    else:
        cfg.association_station_files = {"full": cfg.full_sta_file}
    cfg.subnet_sta_files = list(cfg.association_station_files.values())
    cfg.in_dir = args.in_dir
    cfg.in_glob = args.in_glob
    cfg.num_workers = args.num_workers
    cfg.out_monitoring_dir = os.path.join(cfg.out_root, "monitoring")
    legacy_monitoring_dir = os.path.join(cfg.out_root, "timing")
    if (
        os.path.isdir(legacy_monitoring_dir)
        and not os.path.exists(cfg.out_monitoring_dir)
    ):
        try:
            os.replace(legacy_monitoring_dir, cfg.out_monitoring_dir)
            print(
                "migrated legacy monitoring directory: {} -> {}".format(
                    legacy_monitoring_dir, cfg.out_monitoring_dir
                ),
                flush=True,
            )
        except OSError as exc:
            print(
                "warning: could not migrate legacy timing directory {}; "
                "new monitoring records will use {} | {}".format(
                    legacy_monitoring_dir, cfg.out_monitoring_dir, exc
                ),
                flush=True,
            )
    elif os.path.isdir(legacy_monitoring_dir):
        print(
            "legacy timing directory remains beside active monitoring "
            "directory: {}".format(legacy_monitoring_dir),
            flush=True,
        )
    cfg.done_record_path = os.path.join(cfg.out_root, "done_record_list.txt")
    cfg.bad_record_path = os.path.join(cfg.out_root, "bad_record_list.csv")
    cfg.timing_plot_script = os.path.join(BASE_DIR, "plot_realtime_timing.py")

    cfg.native_picker_settings = json.loads(args.native_pickers_json)
    cfg.reference_picker_settings = json.loads(args.reference_pickers_json)
    cfg.repicker_pos_neg_settings = json.loads(args.repicker_pos_neg_json)
    cfg.repicker_pos_settings = json.loads(args.repicker_pos_json)
    duplicate_names = (
        set(cfg.native_picker_settings) & set(cfg.reference_picker_settings)
    )
    if duplicate_names:
        raise ValueError("pickers cannot be native and reference: {}".format(
            sorted(duplicate_names)
        ))
    picker_pos_neg = list(dict.fromkeys(cfg.picker_pos_neg_group))
    cfg.reference_workflow_specs = reference_workflows(cfg)
    cfg.reference_picker_names = list(dict.fromkeys(
        item["picker"] for item in cfg.reference_workflow_specs.values()
    ))
    references = set(cfg.reference_picker_names)
    reference_branches = list(cfg.reference_workflow_specs)
    cfg.reference_associator_settings = json.loads(
        getattr(args, "reference_associators_json", "{}")
    )
    if any(item["associator"] == "GaMMA" for item in cfg.reference_workflow_specs.values()):
        for name, workflow in cfg.reference_workflow_specs.items():
            if workflow["associator"] == "GaMMA" and not cfg.reference_associator_settings.get(name):
                raise ValueError("missing GaMMA settings for reference workflow: {}".format(name))
        require_gamma_runtime()
    cfg.continuous_picker_groups = {"POS_NEG": picker_pos_neg}
    cfg.continuous_picker_specs = {
        name: {"group": "POS_NEG", "model": name} for name in picker_pos_neg
    }
    if not cfg.continuous_picker_specs:
        raise ValueError("at least one preferred continuous picker is required")
    cfg.enabled_pickers = _enabled_picker_names(cfg)
    configured_plot_ref = list(getattr(
        cfg, "enable_event_waveform_plot_ref", []
    )) if reference_branches else []
    invalid_plot_ref = [
        index for index in configured_plot_ref
        if not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(reference_branches)
    ]
    if invalid_plot_ref:
        raise ValueError(
            "enable_event_waveform_plot_ref contains invalid reference "
            "indexes {} for {}".format(invalid_plot_ref, reference_branches)
        )
    cfg.event_waveform_plot_ref_branches = [
        reference_branches[index] for index in configured_plot_ref
    ]
    missing_native = set(picker_pos_neg) - set(cfg.native_picker_settings)
    missing_reference = references - set(cfg.reference_picker_settings)
    if missing_native or missing_reference:
        raise ValueError(
            "selected picker specifications are missing: POS_NEG={} "
            "reference={}".format(
                sorted(missing_native),
                sorted(missing_reference)
            )
        )
    cfg.continuous_picker_group_min_support = {
        "POS_NEG": int(cfg.picker_pos_neg_group_min_picker_support),
    }
    for group_name, members in cfg.continuous_picker_groups.items():
        if not members:
            print(
                "{} continuous picker group disabled".format(group_name),
                flush=True,
            )
            continue
        required = cfg.continuous_picker_group_min_support[group_name]
        if required <= 0 or required > len(members):
            raise ValueError(
                "{} continuous picker support {} is invalid for {} models"
                .format(group_name, required, len(members))
            )
    cfg.picker_selection_signature = json.dumps(
        {
            "continuous_picker_groups": cfg.continuous_picker_groups,
            "continuous_picker_group_min_support": (
                cfg.continuous_picker_group_min_support
            ),
            "reference_workflows": cfg.reference_workflow_specs,
            "reference_associators": cfg.reference_associator_settings,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cfg.picker_selection_state_dir = os.path.join(
        cfg.out_root, "_internal", "pipeline_state"
    )
    for name in sorted(picker_pos_neg):
        ckpt_path = cfg.native_picker_settings[name].get("ckpt", "")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "{} checkpoint file not found: {}".format(name, ckpt_path)
            )
    cfg.picker_gpu_indices = {}
    for runtime_name in cfg.enabled_pickers:
        continuous_spec = cfg.continuous_picker_specs.get(runtime_name)
        if continuous_spec is None:
            settings = cfg.reference_picker_settings[runtime_name]
        else:
            settings = cfg.native_picker_settings[continuous_spec["model"]]
        cfg.picker_gpu_indices[runtime_name] = int(settings["gpu_idx"])
    print("picker GPU assignments: {}".format(cfg.picker_gpu_indices))

    selected_pos_neg = list(cfg.repicker_pos_neg_group)
    selected_pos = list(cfg.repicker_pos_group)
    missing_pos_neg = set(selected_pos_neg) - set(
        cfg.repicker_pos_neg_settings
    )
    missing_pos = set(selected_pos) - set(cfg.repicker_pos_settings)
    if missing_pos_neg or missing_pos:
        raise ValueError(
            "selected repicker specifications are missing: POS_NEG={} POS={}"
            .format(sorted(missing_pos_neg), sorted(missing_pos))
        )
    if set(picker_pos_neg) - set(selected_pos_neg):
        raise ValueError(
            "continuous POS_NEG pickers must be selected from "
            "repicker_pos_neg_group: {}".format(
                sorted(set(picker_pos_neg) - set(selected_pos_neg))
            )
        )
    cfg.repicker_pos_neg_settings = {
        name: cfg.repicker_pos_neg_settings[name] for name in selected_pos_neg
    }
    cfg.repicker_pos_settings = {
        name: cfg.repicker_pos_settings[name] for name in selected_pos
    }
    if cfg.enable_post_process:
        if not selected_pos_neg or not selected_pos:
            raise ValueError(
                "event repicking requires both repicker groups"
            )
        for group_name, settings in (
            ("POS_NEG", cfg.repicker_pos_neg_settings),
            ("POS", cfg.repicker_pos_settings),
        ):
            if len(settings) < int(cfg.repick_group_min_picker_support):
                raise ValueError(
                    "{} has fewer models than repick_group_min_picker_support"
                    .format(group_name)
                )
            for name, spec in settings.items():
                # Continuous POS_NEG models are reused and need no second load.
                if group_name == "POS_NEG" and name in picker_pos_neg:
                    continue
                ckpt_path = spec.get("ckpt", "")
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError(
                        "{} {} checkpoint file not found: {}".format(
                            group_name, name, ckpt_path
                        )
                    )
    cfg.event_repick_status_path = os.path.join(
        cfg.out_root, "_internal", "event_repick", "event_repick_status.csv"
    )
    cfg.enable_event_waveform_plot = bool(getattr(
        cfg, "enable_event_waveform_plot", cfg.enable_post_process
    ))
    if cfg.event_waveform_plot_ref_branches and not cfg.enable_post_process:
        raise ValueError(
            "reference event waveform plotting currently requires "
            "enable_post_process=True so filtered segment waveforms are retained"
        )
    cfg.out_event_waveform_final_dir = os.path.join(
        cfg.out_root, "event_waveform_final_AI-PAL"
    )

    visible_names = {}
    preferred_runtime_names = picker_pos_neg
    for index, runtime_name in enumerate(preferred_runtime_names, start=1):
        spec = cfg.continuous_picker_specs[runtime_name]
        visible_names[runtime_name] = (
            "1.1.{}_picks_{}_{}".format(
                index, spec["group"].lower(), spec["model"]
            )
        )
    for index, reference_name in enumerate(cfg.reference_picker_names, start=1):
        reference_label = (
            "PhaseNet-SeisBench"
            if reference_name == "PHN-SB" else reference_name
        )
        visible_names[reference_name] = (
            "1.3.{}_picks_ref_{}".format(index, reference_label)
        )
    cfg.picker_output_dirs = {}
    for name in cfg.enabled_pickers:
        if (
            cfg.save_individual_picker_outputs
            or name not in cfg.continuous_picker_specs
        ):
            cfg.picker_output_dirs[name] = os.path.join(cfg.out_root, visible_names[name])
        else:
            cfg.picker_output_dirs[name] = os.path.join(
                cfg.out_root, "_internal", "individual_picks", name
            )

    cfg.association_branch_members = _association_branch_members(cfg)
    layouts = {
        "AI-PAL": {
            "result_name": "AI-PAL",
            "pick_dir_name": "1.2_picks_AI-PAL-ENSEMBLE",
            "initial_phase_dir_name": "2.1.0_phase_init_AI-PAL",
            "phase_dir_name": "2.1_phase_AI-PAL",
            "final_phase_dir_name": "3.1_phase_final_AI-PAL",
        },
    }
    for index, (name, workflow) in enumerate(cfg.reference_workflow_specs.items(), start=1):
        result_name = "{}_{}".format(workflow["picker"], workflow["associator"])
        layouts[name] = {
            "result_name": result_name,
            "pick_dir_name": visible_names[workflow["picker"]],
            "phase_dir_name": "2.2.{}_phase_ref_{}".format(index, result_name),
            "final_phase_dir_name": "3.2.{}_phase_final_ref_{}".format(index, result_name),
        }
    cfg.result_branches = {}
    for branch_name, members in cfg.association_branch_members.items():
        layout = layouts[branch_name]
        result_name = layout["result_name"]
        internal_root = os.path.join(cfg.out_root, "_internal", result_name)
        final_cfg = copy.copy(cfg)
        final_cfg.out_pha_dir = os.path.join(cfg.out_root, layout["phase_dir_name"])
        final_cfg.out_final_pha_dir = os.path.join(
            cfg.out_root, layout["final_phase_dir_name"]
        )
        final_cfg.out_final_ctlg_dir = os.path.join(
            internal_root, "unused_catalog_final"
        )
        final_cfg.out_final_merge_dir = os.path.join(internal_root, "final_merge")
        final_cfg.segment_window_path = os.path.join(
            final_cfg.out_final_merge_dir, "segment_windows.csv"
        )
        final_cfg.finalized_window_path = os.path.join(
            final_cfg.out_final_merge_dir, "finalized_windows.csv"
        )
        final_cfg.write_catalog_outputs = False
        final_cfg.result_name = result_name
        pick_dir = (
            cfg.picker_output_dirs[members[0]]
            if len(members) == 1
            else os.path.join(cfg.out_root, layout["pick_dir_name"])
        )
        cfg.result_branches[branch_name] = {
            "associator": cfg.reference_workflow_specs.get(branch_name, {}).get("associator", "PAL"),
            "associator_config": cfg.reference_associator_settings.get(branch_name, {}),
            "result_name": result_name,
            "members": members,
            "pick_dir": pick_dir,
            "initial_phase_dir": (
                os.path.join(cfg.out_root, layout["initial_phase_dir_name"])
                if layout.get("initial_phase_dir_name") else None
            ),
            "phase_dir": final_cfg.out_pha_dir,
            "subnet_phase_dir": os.path.join(internal_root, "subnet_phase"),
            "merge_dir": os.path.join(internal_root, "subnet_merge"),
            "final_cfg": final_cfg,
        }
    cfg.out_event_waveform_final_ref_dirs = {
        branch_name: os.path.join(
            cfg.out_root,
            "event_waveform_final_ref_{}".format(
                cfg.result_branches[branch_name]["result_name"]
            ),
        )
        for branch_name in cfg.event_waveform_plot_ref_branches
    }
    cfg.association_branches = list(cfg.result_branches)
    return cfg

def subnet_name_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("station_realtime_scsn_complete_", "")


def get_assoc_param(cfg, subnet_name, param):
    configured = getattr(cfg, "subnet_assoc_params", {})
    params = dict(configured.get("default", {}))
    params.update(configured.get(subnet_name, {}))
    suffix = subnet_name.split("_")[-1]
    if suffix != subnet_name:
        params.update(configured.get(suffix, {}))
    if param not in params:
        if param in ("lat_range", "lon_range"):
            return None
        raise KeyError("Missing {} association parameter for {}".format(
            param, subnet_name
        ))
    return params[param]


class ParallelSubnetAssociators(object):
    def __init__(self, cfg):
        startup_t0 = time.perf_counter()
        self.closed = False
        self.associators = {}
        self.station_dicts = {}
        self.pick_sta_dict = {}
        self.startup_timing = {}
        self.full_station_dict = rtp.get_realtime_sta_dict(cfg.full_sta_file)
        specifications = {}
        for subnet_name, fsta in cfg.association_station_files.items():
            sta_dict = rtp.get_realtime_sta_dict(fsta)
            self.pick_sta_dict.update(sta_dict)
            specifications[subnet_name] = (sta_dict, {
                param: get_assoc_param(cfg, subnet_name, param)
                for param in (
                    "xy_margin", "xy_grid", "z_grids", "min_sta",
                    "ot_dev", "max_res", "max_drop", "vp",
                    "lat_range", "lon_range",
                )
            })
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, len(specifications)),
            thread_name_prefix="PAL-subnet",
        )
        futures = {
            subnet_name: self.executor.submit(
                self._build_associator, sta_dict, assoc_params
            )
            for subnet_name, (sta_dict, assoc_params) in specifications.items()
        }
        for subnet_name, future in futures.items():
            associator, elapsed = future.result()
            sta_dict = specifications[subnet_name][0]
            self.associators[subnet_name] = associator
            self.station_dicts[subnet_name] = sta_dict
            result = {
                "subnet": subnet_name,
                "time_table_sec": elapsed,
                "num_stations": len(sta_dict),
            }
            self.startup_timing[subnet_name] = result
            print(
                "{} time table: {:.2f}s ({} station selectors)".format(
                    subnet_name, result["time_table_sec"], result["num_stations"]
                )
            )
        self.time_table_wall_sec = time.perf_counter() - startup_t0
        print("parallel time-table wall time: {:.2f}s".format(
            self.time_table_wall_sec
        ))
        print("initialized {} threaded association network(s)".format(
            len(self.associators)
        ))

    @staticmethod
    def _build_associator(sta_dict, assoc_params):
        started = time.perf_counter()
        associator = associator_pal.PS_Pair_Assoc(sta_dict, **assoc_params)
        return associator, time.perf_counter() - started

    def assert_healthy(self):
        if self.closed:
            raise RuntimeError("PAL association thread pool is closed")

    def health_summary(self):
        return ";".join(
            "{}:thread:ready".format(name)
            for name in sorted(self.associators)
        )

    def _associate_one(self, subnet_name, picks, pha_path):
        sta_dict = self.station_dicts[subnet_name]
        subnet_picks = picks[[pick["net_sta"] in sta_dict for pick in picks]]
        num_subnet_picks = len(subnet_picks)
        started = time.perf_counter()
        try:
            with open(os.devnull, "w") as out_ctlg, open(pha_path, "w") as out_pha:
                self.associators[subnet_name].associate(
                    subnet_picks, out_ctlg, out_pha
                )
            num_prob_filled = rtp.enrich_phase_probabilities(
                pha_path, subnet_picks
            )
        finally:
            del subnet_picks
            # Association grids are allocated in this persistent worker
            # thread, so trim from the same worker after every task.
            rtp.trim_cpu_allocator()
        return {
            "type": "associated",
            "subnet": subnet_name,
            "assoc_sec": time.perf_counter() - started,
            "num_picks": num_subnet_picks,
            "num_prob_filled": num_prob_filled,
            "pha_path": pha_path,
        }

    def associate(self, picks, segment, result_name, branch):
        self.assert_healthy()
        if branch.get("associator", "PAL") == "GaMMA":
            return run_gamma(picks, self.full_station_dict, segment, branch)
        futures = {
            subnet_name: self.executor.submit(
                self._associate_one,
                subnet_name,
                picks,
                os.path.join(
                    branch["subnet_phase_dir"],
                    "phase_{}_{}_{}.dat".format(
                        segment, result_name, subnet_name
                    ),
                ),
            )
            for subnet_name in self.associators
        }
        return {
            subnet_name: future.result()
            for subnet_name, future in futures.items()
        }

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)

if __name__ == "__main__":
    cfg, args = parse_args()
    cfg = apply_overrides(cfg, args)
    runtime_console.configure(cfg)
    configure_torch_backends(cfg)
    runtime_console.log(
        "AI-PAL",
        "realtime startup | continuous={} | reference={} | postprocess={}".format(
            cfg.continuous_picker_groups,
            cfg.reference_picker_names,
            "enabled" if cfg.enable_post_process else "disabled",
        ),
    )
    print("AI-PAL config source: {}".format(os.path.abspath(config_ai_pal.__file__)))
    print("PAL associator source: {}".format(
        os.path.abspath(associator_pal.__file__)
    ))

    pick_sta_dict = rtp.get_realtime_sta_dict(cfg.full_sta_file)
    print("loaded {} station selectors for picking from full station file".format(
        len(pick_sta_dict)
    ))
    print("association station sets: {}".format(
        sorted(cfg.association_station_files)
    ))
    for branch_name in cfg.association_branches:
        if (
            cfg.enable_post_process
            and branch_name == rtp.PREFERRED_ENSEMBLE_BRANCH
        ):
            # Preferred-branch finalization must wait until existing segment
            # phase files have been repicked and reassociated below.
            continue
        try:
            print("startup final merge for {}".format(branch_name), flush=True)
            rtp.bootstrap_final_merged_outputs(
                cfg.result_branches[branch_name]["final_cfg"]
            )
        except Exception as exc:
            print("warning: startup final time merge failed for {} | {}: {}".format(
                branch_name, exc.__class__.__name__, exc
            ))

    native_registry = {
        "SAR": ("picker_SAR.picker", "SAR_Picker"),
        "FT": ("picker_FT.picker", "FT_Picker"),
        "PHN": ("picker_PHN.picker", "PHN_Picker"),
        "RUN": ("picker_RUN.picker", "RUN_Picker"),
    }
    pickers = {}
    model_load_times = {}
    runtime_console.log(
        "startup",
        "loading {} continuous/reference models | devices={}".format(
            len(cfg.enabled_pickers), cfg.picker_gpu_indices
        ),
    )
    for picker_name in cfg.enabled_pickers:
        t0 = time.perf_counter()
        continuous_spec = cfg.continuous_picker_specs.get(picker_name)
        if continuous_spec is not None:
            model_name = continuous_spec["model"]
            settings = cfg.native_picker_settings[model_name]
            module_name, class_name = native_registry[model_name]
            module = importlib.import_module(module_name)
            picker_class = getattr(module, class_name)
            native_picker = picker_class(
                settings["ckpt"],
                -1,
                int(settings["gpu_idx"]),
            )
            pickers[picker_name] = realtime_pickers.NativePickerAdapter(
                picker_name, native_picker
            )
        elif picker_name == "PHN-SB":
            settings = cfg.reference_picker_settings[picker_name]
            reference_cfg = _load_picker_config(
                settings["config"], "ai_pal_phn_sb_config"
            )
            pickers[picker_name] = realtime_pickers.SeisBenchPhaseNetPicker(
                reference_cfg, int(settings["gpu_idx"])
            )
        model_load_times[picker_name] = time.perf_counter() - t0
        print("{} model load time: {:.2f}s".format(
            picker_name, model_load_times[picker_name]
        ))
    inference_executors = rtp.RealtimeInferenceExecutors(
        pickers, cfg.num_workers
    )
    subnet_associators = ParallelSubnetAssociators(cfg)
    event_repick_coordinator = None
    if cfg.enable_post_process:
        from event_repicker import EventRepicker, RealtimeEventRepickCoordinator
        event_repicker = EventRepicker(
            AI_PAL_ROOT,
            cfg,
            cfg.repicker_pos_neg_settings,
            cfg.full_sta_file,
            station_dict=pick_sta_dict,
            repicker_pos_specs=cfg.repicker_pos_settings,
            shared_continuous_pickers={
                group_name: {
                    cfg.continuous_picker_specs[runtime_name]["model"]: (
                        pickers[runtime_name]
                    )
                    for runtime_name in runtime_names
                    if runtime_name in pickers
                }
                for group_name, runtime_names
                in cfg.continuous_picker_groups.items()
            },
        )
        event_repicker.load_pickers()
        print(
            "dual repicker groups preloaded before waveform ingest | RSS {:.1f} MB"
            .format(rtp.release_transient_memory()),
            flush=True,
        )
        event_repick_coordinator = RealtimeEventRepickCoordinator(
            event_repicker,
            cfg.event_repick_status_path,
        )
        print(
            "realtime event repicking enabled for preferred branch {}".format(
                rtp.PREFERRED_ENSEMBLE_BRANCH
            ),
            flush=True,
        )
        branch_name = rtp.PREFERRED_ENSEMBLE_BRANCH
        branch_cfg = cfg.result_branches[branch_name]["final_cfg"]
        pending_repick = rtp.pending_segment_repick_results(
            branch_cfg, event_repick_coordinator.completed_phase_paths
        )
        print(
            "startup segment repick supplement for {}: {} original segments"
            .format(branch_name, len(pending_repick)),
            flush=True,
        )
        input_by_segment = {
            rtp.segment_code(path): path
            for path in glob.glob(os.path.join(cfg.in_dir, cfg.in_glob))
        }
        for result in pending_repick:
            segment = result["segment"]
            mseed_path = input_by_segment.get(segment)
            if mseed_path is None:
                print(
                    "warning: startup segment repick missing MiniSEED for {}"
                    .format(segment),
                    flush=True,
                )
                continue
            event_repick_coordinator.begin_segment(segment)
            print(
                "startup segment repick preprocessing {}".format(mseed_path),
                flush=True,
            )
            try:
                waveforms, waveform_timing = (
                    rtp.prepare_realtime_segment_waveforms(
                        mseed_path, pick_sta_dict, cfg
                    )
                )
            except Exception as exc:
                print(
                    "warning: startup segment repick cannot prepare {} | "
                    "{}: {}".format(
                        mseed_path, exc.__class__.__name__, exc
                    ),
                    flush=True,
                )
                continue
            event_repick_coordinator.register_segment(
                segment,
                waveform_timing["data_start"],
                waveform_timing["data_end"],
                waveforms,
            )
            event_repick_coordinator.process_segment_result(
                branch_name, segment, result["phase_path"]
            )
        event_repick_coordinator.release_segment_cache(
            "after startup segment repicking"
        )
        print(
            "startup final merge for repicked {}".format(branch_name),
            flush=True,
        )
        try:
            startup_final_results = rtp.bootstrap_final_merged_outputs(
                branch_cfg,
                eligible_phase_paths=(
                    event_repick_coordinator.completed_phase_paths
                ),
            )
            event_repick_coordinator.plot_final_results(
                startup_final_results
            )
        except Exception as exc:
            print(
                "warning: startup final time merge failed for {} | {}: {}"
                .format(branch_name, exc.__class__.__name__, exc),
                flush=True,
            )
    runtime_console.log(
        "ready",
        "stations={} | association_networks={} | initialization complete".format(
            len(pick_sta_dict), len(subnet_associators.associators)
        ),
    )
    try:
        rtp.write_initialization_timing(
            cfg.out_monitoring_dir,
            subnet_associators.startup_timing,
            model_load_times=model_load_times,
            time_table_wall_sec=subnet_associators.time_table_wall_sec,
        )
        rtp.realtime_loop(
            pickers,
            subnet_associators,
            pick_sta_dict,
            cfg,
            event_repick_coordinator=event_repick_coordinator,
            inference_executors=inference_executors,
        )
    finally:
        if event_repick_coordinator is not None:
            event_repick_coordinator.close()
        inference_executors.close()
        for picker in pickers.values():
            close_picker = getattr(picker, "close", None)
            if close_picker is not None:
                close_picker()
        subnet_associators.close()
