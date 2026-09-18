"""Launch pseudo-realtime multi-picker inference and PAL association."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


# ============================================================================
# USER SETTINGS: I/O PATHS
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # Packaged example; drives configs and local output paths.
# Config paths are derived from CASE_CODE; picker selection stays explicit.
CONFIG_AI_PAL = Path("config_ai_pal_%s.py" % CASE_CODE)
# Set to None to pick the selector-deduplicated union of all subnet files.
FULL_STATION_FILE = None

# Leave empty to associate a provided FULL_STATION_FILE once with "full".
# Otherwise these map in order to r1, r2, ... and only subnets are associated;
# they are also the required picking-union inputs when the full file is None.
SUBNET_STATION_FILES = [
    "input/station_scedc_realtime_r1_ai-pal-v1.csv",
    "input/station_scedc_realtime_r2_ai-pal-v1.csv",
    "input/station_scedc_realtime_r3_ai-pal-v1.csv",
    "input/station_scedc_realtime_r4_ai-pal-v1.csv",
    "input/station_scedc_realtime_r5_ai-pal-v1.csv",
    "input/station_scedc_realtime_r6_ai-pal-v1.csv",
]
IN_DIR = "/app/aqms/ai_pal/IN"
IN_GLOB = "*.ms"
OUT_ROOT = "output/%s_realtime" % CASE_CODE


# ============================================================================
# USER SETTINGS: PICKERS, DEVICES, AND CHECKPOINTS
# ============================================================================
# Available POS_NEG picker specifications. Select continuous models with
# picker_pos_neg_group in config_ai_pal_<case>.py. Set gpu_idx=-1 for CPU.
PICKERS_POS_NEG = {
    "SAR": {
        "config": Path("config_sar_%s.py" % CASE_CODE),
        "gpu_idx": 0,
        "ckpt": Path("input/SoCal_ckpt/sar_best.ckpt"),
    },
    "FT": {
        "config": Path("config_ft_%s.py" % CASE_CODE),
        "gpu_idx": 1,
        "ckpt": Path("input/SoCal_ckpt/ft_best.ckpt"),
    },
    "PHN": {
        "config": Path("config_phn_%s.py" % CASE_CODE),
        "gpu_idx": 2,
        "ckpt": Path("input/SoCal_ckpt/phn_best.ckpt"),
    },
    "RUN": {
        "config": Path("config_run_%s.py" % CASE_CODE),
        "gpu_idx": 3,
        "ckpt": Path("input/SoCal_ckpt/run_best.ckpt"),
    },
}

# Positive-only models, selected by repicker_pos_group for post-processing.
PICKERS_POS = {
    "SAR": {
        "config": Path("config_sar_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_sar_best.ckpt"),
    },
    "FT": {
        "config": Path("config_ft_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_ft_best.ckpt"),
    },
    "PHN": {
        "config": Path("config_phn_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_phn_best.ckpt"),
    },
    "RUN": {
        "config": Path("config_run_pos_ceed.py"),
        "gpu_idx": -1,
        "ckpt": Path("input/CEED_ckpt/ceed_pos_run_best.ckpt"),
    },
}

# Available reference-picker specifications. Select combinations in reference_workflows
# in config_ai_pal_<case>.py. Reference picks are shared across associators
# and are never added to the preferred ensemble.
PICKER_REF = {
    "PHN-SB": {
        "config": Path("config_ref_phn-sb_%s.py" % CASE_CODE),
        "gpu_idx": -1,
    },
}

ASSOCIATORS_REF = {
    "GaMMA": {"config": Path("config_ref_gamma_%s.py" % CASE_CODE)},
}  # Reference PAL uses the PAL parameters in CONFIG_AI_PAL.

# Number of stations prepared concurrently. Inference is serialized per device.
NUM_WORKERS = 5


# ============================================================================
# CONNECTION CODE: NORMALLY NO USER EDITS BELOW THIS LINE
# ============================================================================
PAL_SRC = AI_PAL_ROOT / "PAL_src"
NATIVE_CONFIG_TARGETS = {
    "SAR": AI_PAL_ROOT / "picker_SAR" / "config.py",
    "FT": AI_PAL_ROOT / "picker_FT" / "config.py",
    "PHN": AI_PAL_ROOT / "picker_PHN" / "config.py",
    "RUN": AI_PAL_ROOT / "picker_RUN" / "config.py",
}

if not CONFIG_AI_PAL.is_file():
    raise FileNotFoundError(CONFIG_AI_PAL)
if str(PAL_SRC) not in sys.path:
    sys.path.insert(0, str(PAL_SRC))
config_spec = importlib.util.spec_from_file_location(
    "ai_pal_realtime_case_config", CONFIG_AI_PAL
)
if config_spec is None or config_spec.loader is None:
    raise ImportError("cannot load {}".format(CONFIG_AI_PAL))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
selection_cfg = config_module.Config()
picker_pos_neg_names = list(dict.fromkeys(
    selection_cfg.picker_pos_neg_group
))
from reference_association import reference_workflows, load_reference_configs
reference_names = list(dict.fromkeys(
    item["picker"] for item in reference_workflows(selection_cfg).values()
))
reference_associators = load_reference_configs(
    selection_cfg, Path.cwd(), ASSOCIATORS_REF
)
if not picker_pos_neg_names:
    raise ValueError("at least one preferred continuous picker is required")
pos_neg_names = list(dict.fromkeys(selection_cfg.repicker_pos_neg_group))
pos_names = list(dict.fromkeys(selection_cfg.repicker_pos_group))
if set(picker_pos_neg_names) - set(pos_neg_names):
    raise ValueError(
        "picker_pos_neg_group must be a subset of repicker_pos_neg_group"
    )
missing = {
    "PICKERS_POS_NEG": sorted(
        set(picker_pos_neg_names) - set(PICKERS_POS_NEG)
    ),
    "PICKER_REF": sorted(set(reference_names) - set(PICKER_REF)),
    "PICKERS_POS_NEG (post-processing)": sorted(
        set(pos_neg_names) - set(PICKERS_POS_NEG)
    ),
    "PICKERS_POS (post-processing)": sorted(
        set(pos_names) - set(PICKERS_POS)
    ),
}
missing = {key: names for key, names in missing.items() if names}
if missing:
    raise ValueError("selected picker specifications are missing: {}".format(missing))

selected_pickers = {
    name: PICKERS_POS_NEG[name] for name in picker_pos_neg_names
}
selected_references = {name: PICKER_REF[name] for name in reference_names}
selected_pos_neg = {
    name: PICKERS_POS_NEG[name] for name in pos_neg_names
}
selected_pos = {name: PICKERS_POS[name] for name in pos_names}

for settings in (
    list(selected_pickers.values()) + list(selected_pos_neg.values())
    + list(selected_pos.values())
):
    for key in ("config", "ckpt"):
        path = Path(settings[key])
        if not path.is_file():
            raise FileNotFoundError(path)
for settings in selected_references.values():
    path = Path(settings["config"])
    if not path.is_file():
        raise FileNotFoundError(path)

shutil.copyfile(CONFIG_AI_PAL, PAL_SRC / "config_ai_pal.py")
for name, settings in selected_pickers.items():
    if name not in NATIVE_CONFIG_TARGETS:
        raise KeyError("unsupported native picker: {}".format(name))
    shutil.copyfile(settings["config"], NATIVE_CONFIG_TARGETS[name])

native_runtime = {
    name: {
        "gpu_idx": int(settings["gpu_idx"]),
        "ckpt": str(Path(settings["ckpt"]).resolve()),
    }
    for name, settings in selected_pickers.items()
}
reference_runtime = {
    name: {
        "gpu_idx": int(settings["gpu_idx"]),
        "config": str(Path(settings["config"]).resolve()),
    }
    for name, settings in selected_references.items()
}
pos_neg_runtime = {
    name: {
        "gpu_idx": int(settings["gpu_idx"]),
        "config": str(Path(settings["config"]).resolve()),
        "ckpt": str(Path(settings["ckpt"]).resolve()),
    }
    for name, settings in selected_pos_neg.items()
}
pos_runtime = {
    name: {
        "gpu_idx": int(settings["gpu_idx"]),
        "config": str(Path(settings["config"]).resolve()),
        "ckpt": str(Path(settings["ckpt"]).resolve()),
    }
    for name, settings in selected_pos.items()
}

print(
    "selected pickers: continuous POS_NEG={} | reference={} | "
    "repickers POS_NEG={} POS={}".format(
        picker_pos_neg_names, reference_names,
        pos_neg_names, pos_names
    ),
    flush=True,
)

command = [
    sys.executable,
    "-u",
    str(PAL_SRC / "run_realtime.py"),
    "--subnet_sta_files",
] + SUBNET_STATION_FILES + [
    "--in_dir={}".format(IN_DIR),
    "--in_glob={}".format(IN_GLOB),
    "--out_root={}".format(OUT_ROOT),
    "--num_workers={}".format(NUM_WORKERS),
    "--native_pickers_json={}".format(json.dumps(native_runtime)),
    "--reference_pickers_json={}".format(json.dumps(reference_runtime)),
    "--reference_associators_json={}".format(json.dumps(reference_associators)),
    "--repicker_pos_neg_json={}".format(json.dumps(pos_neg_runtime)),
    "--repicker_pos_json={}".format(json.dumps(pos_runtime)),
]
if FULL_STATION_FILE is not None:
    command.extend(["--full_sta_file", str(FULL_STATION_FILE)])
print("launching realtime pipeline with PID-owning parent {}".format(os.getpid()))
try:
    subprocess.check_call(command)
except subprocess.CalledProcessError as exc:
    monitoring_dir = os.path.join(OUT_ROOT, "monitoring")
    os.makedirs(monitoring_dir, exist_ok=True)
    with open(
        os.path.join(monitoring_dir, "realtime_launcher_exit.log"), "a"
    ) as fp:
        fp.write("{} pid={} returncode={}\n".format(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            os.getpid(), exc.returncode,
        ))
        fp.flush()
        os.fsync(fp.fileno())
    print(
        "realtime pipeline exited unexpectedly with return code {}".format(
            exc.returncode
        ),
        flush=True,
    )
    if exc.returncode == -9:
        print(
            "SIGKILL (9): inspect monitoring/memory_stages_<segment>.csv and "
            "realtime_heartbeat.csv. The stage file records both process RSS "
            "and total cgroup memory against its limit, including memory held "
            "by child processes and the batch scheduler.",
            flush=True,
        )
    raise
