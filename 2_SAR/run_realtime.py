"""Run pseudo-realtime multi-model picking and PAL association.

This script is intended to be launched from ``run_sar/4_pick_assoc_realtime_eg.py``
after copying a realtime config into ``2_SAR/config.py``.
"""
import argparse
import copy
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "1_PAL"))
if PAL_DIR not in sys.path:
    sys.path.append(PAL_DIR)

import associator_pal
import config
import picker
import realtime_pipeline as rtp
import realtime_pickers


def parse_args():
    cfg = config.Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subnet_sta_files", nargs="+", required=True)
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--in_glob", type=str, default="*.ms")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_idx", type=int, default=-1)
    parser.add_argument("--pg1_gpu_idx", type=int, default=0)
    parser.add_argument("--pg2_gpu_idx", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return cfg, parser.parse_args()


def _enabled_picker_names(cfg):
    names = []
    for group in cfg.picker_groups:
        for name in group:
            if name not in names:
                names.append(name)
    supported = {"SAR", "PHN-SB"}
    unknown = [name for name in names if name not in supported]
    if unknown:
        raise ValueError("unsupported realtime pickers: {}".format(unknown))
    return names


def apply_overrides(cfg, args):
    if cfg.association_methods != ["PAL"]:
        raise ValueError("currently supported association_methods: ['PAL']")
    cfg.subnet_sta_files = args.subnet_sta_files
    cfg.in_dir = args.in_dir
    cfg.in_glob = args.in_glob
    cfg.out_root = args.out_root
    cfg.ckpt_dir = args.ckpt_dir
    cfg.ckpt_idx = args.ckpt_idx
    if len(cfg.picker_groups) > 2:
        raise ValueError("add GPU controls before configuring more than two picker groups")
    group_gpu_indices = [args.pg1_gpu_idx, args.pg2_gpu_idx]
    cfg.picker_gpu_indices = {}
    for group_idx, picker_group in enumerate(cfg.picker_groups):
        for picker_name in picker_group:
            cfg.picker_gpu_indices[picker_name] = group_gpu_indices[group_idx]
    cfg.num_workers = args.num_workers
    cfg.out_timing_dir = os.path.join(cfg.out_root, "timing")
    cfg.done_record_path = os.path.join(cfg.out_root, "done_record_list.txt")
    cfg.bad_record_path = os.path.join(cfg.out_root, "bad_record_list.csv")
    cfg.timing_plot_script = os.path.join(BASE_DIR, "plot_realtime_timing.py")

    layouts = {
        "SAR": {
            "result_name": "SAR_PAL",
            "pick_dir_name": "1.1_picks_SAR",
            "phase_dir_name": "2.1_phase_SAR_PAL",
            "final_phase_dir_name": "3.1_phase_final_SAR_PAL",
        },
        "PHN-SB": {
            "result_name": "PHN-SB_PAL",
            "pick_dir_name": "1.2_picks_PhaseNet-SeisBench",
            "phase_dir_name": "2.2_phase_PHN-SB_PAL",
            "final_phase_dir_name": "3.2_phase_final_PHN-SB_PAL",
        },
    }
    cfg.enabled_pickers = _enabled_picker_names(cfg)
    print("picker-group GPU assignments: {}".format(cfg.picker_gpu_indices))
    cfg.result_branches = {}
    for picker_name in cfg.enabled_pickers:
        layout = layouts[picker_name]
        result_name = layout["result_name"]
        internal_root = os.path.join(cfg.out_root, "_internal", result_name)
        final_cfg = copy.copy(cfg)
        final_cfg.out_pha_dir = os.path.join(
            cfg.out_root, layout["phase_dir_name"]
        )
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
        cfg.result_branches[picker_name] = {
            "result_name": result_name,
            "pick_dir": os.path.join(cfg.out_root, layout["pick_dir_name"]),
            "phase_dir": final_cfg.out_pha_dir,
            "subnet_phase_dir": os.path.join(internal_root, "subnet_phase"),
            "merge_dir": os.path.join(internal_root, "subnet_merge"),
            "final_cfg": final_cfg,
        }
    return cfg

def subnet_name_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("station_realtime_scsn_complete_", "")


def get_assoc_param(cfg, subnet_name, param):
    params = getattr(cfg, "subnet_assoc_params", {})
    subnet_params = params.get(subnet_name, {})
    if not subnet_params:
        subnet_params = params.get(subnet_name.split("_")[-1], {})
    if not subnet_params:
        subnet_params = params.get("default", {})
    if param not in subnet_params:
        raise KeyError("Missing {} association parameter for {}".format(param, subnet_name))
    return subnet_params[param]


def subnet_assoc_worker(subnet_name, fsta, assoc_params, input_queue, result_queue):
    try:
        sta_dict = rtp.get_realtime_sta_dict(fsta)
        t0 = time.perf_counter()
        associator = associator_pal.PS_Pair_Assoc(sta_dict, **assoc_params)
        result_queue.put({
            "type": "ready",
            "subnet": subnet_name,
            "time_table_sec": time.perf_counter() - t0,
            "num_stations": len(sta_dict),
        })

        while True:
            command = input_queue.get()
            if command["type"] == "stop":
                return
            if command["type"] != "associate":
                continue

            picks = command["picks"]
            subnet_picks = picks[[pick["net_sta"] in sta_dict for pick in picks]]
            t0 = time.perf_counter()
            with open(os.devnull, "w") as out_ctlg, \
                    open(command["pha_path"], "w") as out_pha:
                associator.associate(subnet_picks, out_ctlg, out_pha)
            num_prob_filled = rtp.enrich_phase_probabilities(
                command["pha_path"], subnet_picks
            )
            result_queue.put({
                "type": "associated",
                "subnet": subnet_name,
                "assoc_sec": time.perf_counter() - t0,
                "num_picks": len(subnet_picks),
                "num_prob_filled": num_prob_filled,
                "pha_path": command["pha_path"],
            })
    except Exception as exc:
        result_queue.put({
            "type": "error",
            "subnet": subnet_name,
            "error": repr(exc),
        })


class ParallelSubnetAssociators(object):
    def __init__(self, cfg):
        startup_t0 = time.perf_counter()
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.input_queues = {}
        self.processes = {}
        self.pick_sta_dict = {}
        self.startup_timing = {}

        for fsta in cfg.subnet_sta_files:
            subnet_name = subnet_name_from_path(fsta)
            sta_dict = rtp.get_realtime_sta_dict(fsta)
            self.pick_sta_dict.update(sta_dict)
            assoc_params = {
                param: get_assoc_param(cfg, subnet_name, param)
                for param in (
                    "xy_margin", "xy_grid", "z_grids", "min_sta",
                    "ot_dev", "max_res", "max_drop", "vp",
                )
            }
            input_queue = self.ctx.Queue()
            proc = self.ctx.Process(
                target=subnet_assoc_worker,
                args=(subnet_name, fsta, assoc_params, input_queue, self.result_queue),
            )
            proc.start()
            self.input_queues[subnet_name] = input_queue
            self.processes[subnet_name] = proc

        for _ in self.processes:
            result = self.result_queue.get()
            self._raise_if_error(result)
            self.startup_timing[result["subnet"]] = result
            print(
                "{} time table: {:.2f}s ({} station selectors)".format(
                    result["subnet"], result["time_table_sec"], result["num_stations"]
                )
            )
        self.time_table_wall_sec = time.perf_counter() - startup_t0
        print("parallel time-table wall time: {:.2f}s".format(
            self.time_table_wall_sec
        ))
        print("loaded {} unique station selectors for picking".format(
            len(self.pick_sta_dict)
        ))

    def associate(self, picks, segment, result_name, branch):
        for subnet_name, input_queue in self.input_queues.items():
            input_queue.put({
                "type": "associate",
                "picks": picks,
                "pha_path": os.path.join(
                    branch["subnet_phase_dir"],
                    "phase_{}_{}_{}.dat".format(
                        segment, result_name, subnet_name
                    ),
                ),
            })

        results = {}
        for _ in self.processes:
            result = self.result_queue.get()
            self._raise_if_error(result)
            results[result["subnet"]] = result
        return results
    def close(self):
        for input_queue in self.input_queues.values():
            input_queue.put({"type": "stop"})
        for proc in self.processes.values():
            proc.join()

    @staticmethod
    def _raise_if_error(result):
        if result["type"] == "error":
            raise RuntimeError(
                "subnet {} worker failed: {}".format(result["subnet"], result["error"])
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg, args = parse_args()
    cfg = apply_overrides(cfg, args)

    for picker_name in cfg.enabled_pickers:
        try:
            print("startup final merge for {}".format(picker_name), flush=True)
            rtp.bootstrap_final_merged_outputs(
                cfg.result_branches[picker_name]["final_cfg"]
            )
        except Exception as exc:
            print("warning: startup final time merge failed for {} | {}: {}".format(
                picker_name, exc.__class__.__name__, exc
            ))

    pickers = {}
    model_load_times = {}
    if "SAR" in cfg.enabled_pickers:
        t0 = time.perf_counter()
        sar_picker = picker.SAR_Picker(
            cfg.ckpt_dir, cfg.ckpt_idx, cfg.picker_gpu_indices["SAR"]
        )
        pickers["SAR"] = realtime_pickers.SARPickerAdapter(sar_picker)
        model_load_times["SAR"] = time.perf_counter() - t0
        print("SAR model load time: {:.2f}s".format(model_load_times["SAR"]))
    if "PHN-SB" in cfg.enabled_pickers:
        t0 = time.perf_counter()
        pickers["PHN-SB"] = realtime_pickers.SeisBenchPhaseNetPicker(
            cfg, cfg.picker_gpu_indices["PHN-SB"]
        )
        model_load_times["PHN-SB"] = time.perf_counter() - t0
        print("PHN-SB model load time: {:.2f}s".format(
            model_load_times["PHN-SB"]
        ))

    subnet_associators = ParallelSubnetAssociators(cfg)
    try:
        rtp.write_initialization_timing(
            cfg.out_timing_dir,
            subnet_associators.startup_timing,
            model_load_times=model_load_times,
            time_table_wall_sec=subnet_associators.time_table_wall_sec,
        )
        rtp.realtime_loop(
            pickers, subnet_associators, subnet_associators.pick_sta_dict, cfg
        )
    finally:
        subnet_associators.close()
