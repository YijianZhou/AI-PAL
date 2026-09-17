"""Convert one integrated NPY sample inventory into one Zarr dataset."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, MODELS, AND CONVERSION
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "eg"  # "eg" is the packaged example; use e.g. "sc" for SoCal.
NPY_ROOT = Path("/data/bigdata/%s_train-samples_npy" % CASE_CODE)
ZARR_PATH = Path("/data/bigdata/%s_train-samples.zarr" % CASE_CODE)
OVERWRITE_EXISTING_ZARR = False
ENABLED_MODELS = ["SAR", "FT", "PHN", "RUN"]
NUM_WORKERS = 10
CHUNK_SIZE = 256
PREFETCH_FACTOR = 1
COMPRESSOR = "lz4"
LOG_INTERVAL = 100000
WRITE_BATCH_SIZE = 512

# ============================================================================
# CONNECTION CODE: normally no edits are needed below this line
# ============================================================================

shutil.copyfile(
    "config_ai_pal_{}.py".format(CASE_CODE),
    AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
)
subprocess_env = os.environ.copy()
subprocess_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
    str(AI_PAL_ROOT / "PAL_src"),
    subprocess_env.get("PYTHONPATH"),
)))

# SAR and FT use integer frame labels. PHN and RUN use Gaussian soft labels at
# waveform-sample resolution. One representative converter writes each required
# target family; every enabled model trains from the same waveform/Zarr root.
MODELS = {
    "SAR": {
        "src": AI_PAL_ROOT / "picker_SAR",
        "config": "config_sar_{}.py".format(CASE_CODE),
        "converter": "preprocess/npy2zarr.py",
        "target_type": "frame",
    },
    "FT": {
        "src": AI_PAL_ROOT / "picker_FT",
        "config": "config_ft_{}.py".format(CASE_CODE),
        "converter": "preprocess/npy2zarr.py",
        "target_type": "frame",
    },
    "PHN": {
        "src": AI_PAL_ROOT / "picker_PHN",
        "config": "config_phn_{}.py".format(CASE_CODE),
        "converter": "preprocess/npy2zarr.py",
        "target_type": "sample",
    },
    "RUN": {
        "src": AI_PAL_ROOT / "picker_RUN",
        "config": "config_run_{}.py".format(CASE_CODE),
        "converter": "preprocess/npy2zarr.py",
        "target_type": "sample",
    },
}

unknown = set(ENABLED_MODELS) - set(MODELS)
if unknown:
    raise KeyError("unknown models: {}".format(sorted(unknown)))

# Builder priority is independent of ENABLED_MODELS ordering. SAR is required
# when enabled because its training data include negative frame targets.
target_priority = {
    "frame": ("SAR", "FT"),
    "sample": ("PHN", "RUN"),
}
target_builders = {
    target_type: next(model for model in priority if model in ENABLED_MODELS)
    for target_type, priority in target_priority.items()
    if any(model in ENABLED_MODELS for model in priority)
}

print("enabled model target mapping: {}".format(
    {name: MODELS[name]["target_type"] for name in ENABLED_MODELS}
), flush=True)
print("selected target builders: {}".format(target_builders), flush=True)

if not NPY_ROOT.exists():
    raise FileNotFoundError(NPY_ROOT)
if ZARR_PATH.exists():
    if not OVERWRITE_EXISTING_ZARR:
        raise FileExistsError(
            "{} already exists; enable OVERWRITE_EXISTING_ZARR to rebuild it"
            .format(ZARR_PATH)
        )
    shutil.rmtree(ZARR_PATH)
ZARR_PATH.parent.mkdir(parents=True, exist_ok=True)

for target_type in ("frame", "sample"):
    model_name = target_builders.get(target_type)
    if model_name is None:
        continue
    model = MODELS[model_name]
    src_dir = model["src"]
    converter_path = src_dir / model["converter"]
    shutil.copyfile(model["config"], src_dir / "config.py")
    shutil.copyfile(
        "config_ai_pal_{}.py".format(CASE_CODE),
        src_dir / "config_ai_pal.py",
    )
    command = [
        sys.executable, str(converter_path),
        "--npy_root", str(NPY_ROOT),
        "--out_path", str(ZARR_PATH),
        "--num_workers", str(NUM_WORKERS),
        "--chunk_size", str(CHUNK_SIZE),
        "--prefetch_factor", str(PREFETCH_FACTOR),
        "--compressor", COMPRESSOR,
        "--log_interval", str(LOG_INTERVAL),
    ]
    if model_name != "SAR":
        command.extend(["--write_batch_size", str(WRITE_BATCH_SIZE)])
    print("building {} targets with {} in {}".format(
        target_type, model_name, ZARR_PATH
    ), flush=True)
    subprocess.check_call(command, env=subprocess_env)
