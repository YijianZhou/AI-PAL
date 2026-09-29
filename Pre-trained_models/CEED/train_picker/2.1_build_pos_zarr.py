"""Build one shared waveform Zarr with the target families required by enabled models."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

# ============================================================================
# USER SETTINGS: INPUTS, OUTPUTS, MODELS, AND CONVERSION
# ============================================================================
AI_PAL_ROOT = Path("~/software/AI-PAL").expanduser()  # Installed source package.
CASE_CODE = "ceed"
CONFIG_AI_PAL = Path(__file__).resolve().parent.parent / "preprocess/config_ai_pal_ceed.py"
DATASET_SETTINGS_CONFIRMED = False  # Set True only after replacing placeholder paths.
NPY_ROOT = Path("/data/Example_data/%s_train_npy" % CASE_CODE)
OUT_ZARR = Path("/data/Example_data/%s_train-samples.zarr" % CASE_CODE)
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
SCRIPT_DIR = Path(__file__).resolve().parent
if not DATASET_SETTINGS_CONFIRMED:
    raise ValueError("Optional CEED retraining: replace Example_data paths with real datasets, then set DATASET_SETTINGS_CONFIRMED=True.")
sys.path.insert(0, str(AI_PAL_ROOT / "PAL_src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helpers'))
subprocess_env = os.environ.copy()
subprocess_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
    str(AI_PAL_ROOT / "PAL_src"),
    subprocess_env.get("PYTHONPATH"),
)))

shutil.copyfile(
    CONFIG_AI_PAL,
    AI_PAL_ROOT / "PAL_src" / "config_ai_pal.py",
)

# SAR and FT use integer frame labels. PHN and RUN use Gaussian soft labels at
# waveform-sample resolution. One representative converter writes each required
# target family; every enabled model trains from the same waveform/Zarr root.
MODELS = {
    "SAR": {
        "src": AI_PAL_ROOT / "picker_SAR",
        "config": SCRIPT_DIR / "config_sar_global_ceed.py",
        "converter": "preprocess/npy2zarr.py",
        "target_type": "frame",
    },
    "FT": {
        "src": AI_PAL_ROOT / "picker_FT",
        "config": SCRIPT_DIR / "config_ft_global_ceed.py",
        "converter": "preprocess/npy2zarr.py",
        "target_type": "frame",
    },
    "PHN": {
        "src": AI_PAL_ROOT / "picker_PHN",
        "config": SCRIPT_DIR / "config_phn_global_ceed.py",
        "converter": "preprocess/npy2zarr.py",
        "target_type": "sample",
    },
    "RUN": {
        "src": AI_PAL_ROOT / "picker_RUN",
        "config": SCRIPT_DIR / "config_run_global_ceed.py",
        "converter": "preprocess/npy2zarr.py",
        "target_type": "sample",
    },
}

unknown = set(ENABLED_MODELS) - set(MODELS)
if unknown:
    raise KeyError("unknown models: {}".format(sorted(unknown)))

required_indexes = [NPY_ROOT / "train_pos.npy", NPY_ROOT / "valid_pos.npy"]
missing_indexes = [str(path) for path in required_indexes if not path.exists()]
if missing_indexes:
    raise FileNotFoundError(
        "missing positive shard index(es): {}".format(", ".join(missing_indexes))
    )

if OUT_ZARR.exists():
    raise FileExistsError("Choose a new OUT_ZARR; existing stores are never overwritten: {}".format(OUT_ZARR))
OUT_ZARR.parent.mkdir(parents=True, exist_ok=True)

# Builder priority is independent of ENABLED_MODELS ordering. SAR/FT share the
# frame target family, while PHN/RUN share the sample target family.
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

# Build frame targets first. Positive-only mode prevents the source converters
# from opening train_neg.npy/valid_neg.npy; later converters reuse compatible
# waveform arrays.
for target_type in ("frame", "sample"):
    model_name = target_builders.get(target_type)
    if model_name is None:
        continue
    model = MODELS[model_name]
    src_dir = model["src"]
    converter_path = src_dir / model["converter"]
    shutil.copyfile(model["config"], src_dir / "config.py")
    shutil.copyfile(
        CONFIG_AI_PAL,
        src_dir / "config_ai_pal.py",
    )
    command = [
        sys.executable, str(converter_path),
        "--npy_root", str(NPY_ROOT),
        "--out_path", str(OUT_ZARR),
        "--num_workers", str(NUM_WORKERS),
        "--chunk_size", str(CHUNK_SIZE),
        "--prefetch_factor", str(PREFETCH_FACTOR),
        "--compressor", COMPRESSOR,
        "--log_interval", str(LOG_INTERVAL),
        "--sample_types", "positive",
    ]
    if model_name != "SAR":
        command.extend(["--write_batch_size", str(WRITE_BATCH_SIZE)])
    print("building {} targets with {} in {}".format(
        target_type, model_name, OUT_ZARR
    ), flush=True)
    subprocess.check_call(command, env=subprocess_env)
