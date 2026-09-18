# Picker Benchmark Workflow

This directory is an executable work directory for building benchmark datasets,
training CEED positive-only pickers, and comparing arbitrary AI-PAL checkpoints.
The installed source package is expected at `~/software/AI-PAL`.

All retained work-directory Python scripts use editable settings near the top of
the file. They do not expose an `argparse` interface. Internal source-package
trainers and converters may still receive command-line arguments from these
launchers as connection code.

## Directory Layout

```text
benchmark_picker/
  preprocess/                 raw acquisition and inference-dataset preparation
  train_pos/                  CEED positive-picker training workflow
  benchmark_runs.py           checkpoint/version registry shared by all tests
  run_picker_benchmark.py     positive and noise inference
  evaluate_picker_benchmark.py
  evaluate_pos_picker_predictions.py
  evaluate_noise_false_detections.py
  plot_picker_benchmark.py
```

## 1. Dataset Preprocessing

Run these scripts from `benchmark_picker/`, editing each script's settings block
first.

### Download and index raw data

1. `python preprocess/download_ceed_raw.py`
2. `python preprocess/extract_ceed_phase.py`
3. `python preprocess/download_seisbench_raw.py`
4. `python preprocess/download_seisbench_noise_raw.py`
5. `python preprocess/check_seisbench_downloads.py`
6. `python preprocess/check_seisbench_noise_downloads.py`

The SeisBench downloaders redirect cache and temporary files to the configured
large-data root. The checkers validate metadata/waveform row counts and detect
partial files. `run_download_seisbench_raw.sh` is retained as an optional shell
wrapper for environments that need cache variables established before Python
starts.

### Inspect source-window distributions

- `plot_seisbench_window_stats.py` plots raw window length and pre-P buffer.
- `plot_seisbench_available_window_length.py` measures usable duration after
  non-window quality control.

These diagnostics should be inspected before changing fixed window lengths.

### Build fixed inference shards

1. `python preprocess/build_ceed_fixed_window_npy.py`
2. `python preprocess/build_seisbench_fixed_window_npy.py`
3. `python preprocess/build_seisbench_noise_fixed_window_npy.py`
4. `python preprocess/check_fixed_window_npy_outputs.py`
5. `python preprocess/summarize_fixed_window_failures.py`

CEED positive windows currently use 50 s, SeisBench positive windows use 40 s,
and SeisBench noise windows use 50 s. All outputs are 100 Hz, three-channel NPY
shards with P/S reference slots followed by waveform samples. Noise references
are NaN. The checker verifies shard counts, shapes, metadata identity, phase
bounds, and summary consistency. The failure summarizer reports quality-control
and insufficient-window reasons.

`signal_lib.py` is a shared preprocessing helper, not an executable stage.

## 2. CEED Positive-Picker Training

Only CEED is used for positive-only training in this workflow.

1. Generate the base CEED phase file with
   `python preprocess/extract_ceed_phase.py`.
2. Run `python train_pos/0_analyze_ceed_phase_feature_rarity.py` to calculate FMD,
   spatiotemporal seismicity rate, hypocentral distance, station-level
   train/validation assignment, and `num_aug`. It also writes the feature and
   rarity figures.
3. Run `python train_pos/1_cut_ceed_train_npy.py` to preprocess the raw CEED HDF5
   waveforms and write augmented 25 s NPY training shards.
4. Run `python train_pos/2_build_pos_zarr.py` to create one shared Zarr containing
   `positive_data`, `positive_target_frame`, and `positive_target_sample` for
   train and validation splits.
5. Run `python train_pos/3_train_pos_pickers.py` to train any enabled subset of
   SAR, FT, PHN, and RUN sequentially.

With the integrated dataset already built, start directly at step 5:

```bash
cd ~/software/AI-PAL/benchmark_picker
python train_pos/3_train_pos_pickers.py
```

The launcher reads `/data1/zhouyj/CEED_train_pos.zarr` and writes to
`/nas/zhouyj/AI_ckpt/ceed_pos/<MODEL>/`. Both `train/` and `valid/` must contain
nonempty `positive_data` arrays shaped `[N, 3, 2500]`, frame targets shaped
`[N, 246]` for SAR/FT, and sample targets shaped `[N, 3, 2500]` for PHN/RUN.
The FT/RUN architecture reductions do not change these target layouts.

All four models use batch size 128 and full validation every 5000 steps.
FT uses width 256, four heads, five layers, and FFN width 512; RUN uses one
residual block per encoder, bottleneck, and decoder stage. Training starts
from scratch, not from existing checkpoints: select empty output directories.
The launcher checks every enabled output before modifying installed configs.
After a partially completed multi-model run, select only the unfinished models
and use an empty directory for any model that must restart. Do not launch this
config-staging workflow concurrently with other jobs using the same source tree.

The enabled benchmark entries point to these models' `best.ckpt` files and use
new `*_v7` markers to avoid reusing old prediction results. If OUT_CKPT_ROOT is
changed, update their paths in `benchmark_runs.py` too. Benchmark inference and
evaluation still read fixed-window NPY shards under `/nas/zhouyj/AI_datasets`,
not the training Zarr; run them from `benchmark_picker/` so their relative
`output/` paths agree. These Linux dataset paths must exist on the workstation.

Each model checkpoint directory contains TensorBoard logs,
`<model>_training_metrics.csv`, a live positive/negative accuracy figure in
`<model>_training_progress.png`, and loss plus frame accuracy (SAR/FT) or sample
accuracy (PHN/RUN) in `<model>_training_diagnostics.png`. Both figures include
full-range and 5th-95th percentile zoom panels.

Training parameters live in:

```text
train_pos/config_ai_pal_pos.py
train_pos/config_sar_pos.py
train_pos/config_ft_pos.py
train_pos/config_phn_pos.py
train_pos/config_run_pos.py
```

I/O paths remain in the executable scripts. `ceed_data_pipeline.py` is the
import-only HDF5 reading and preprocessing helper used by
`1_cut_ceed_train_npy.py`. The active workflow writes NPY shards directly and
does not create an intermediate SAC dataset.

`2_build_pos_zarr.py` stages the selected model config into the installed source
package and invokes its converter with positive-only mode. `3_train_pos_pickers.py`
does the same before training. These are training operations; benchmark
inference never modifies installed configs.

## 3. Register Model Versions

Edit `benchmark_runs.py`. Each entry requires:

```python
{
    'model': 'PHN',
    'checkpoint': Path('/nas/.../checkpoint_directory_or_file.ckpt'),
    'marker': 'phn_lr2_filters16',
    'label': 'PHN lr=1e-2, filters=16',
    'color': '#CC6677',
}
```

- A checkpoint directory prefers `best.ckpt`; otherwise the installed picker
  selects a numbered checkpoint.
- A checkpoint file selects that exact model.
- `marker` must be unique and filesystem-safe. It propagates to predictions,
  evaluation directories, CSV outputs, and plots.
- `ENABLED_RUN_MARKERS` selects a subset. An empty list enables every entry.
- `COMPARISON_MARKER` names the combined result directory.

The installed `~/software/AI-PAL/picker_<MODEL>/config.py` must describe the
checkpoint architecture. Inference treats it as read-only. A structural mismatch
is caught by strict checkpoint loading.

## 4. Run Inference

Edit runtime controls at the top of `run_picker_benchmark.py`, then run:

```bash
python run_picker_benchmark.py
```

The launcher applies the same random-window ensemble independently to every
event or noise trace. It supports CEED and all configured SeisBench positive
datasets, plus explicit SeisBench noise subsets. Completed outputs with valid
summary JSON files are skipped unless `OVERWRITE_COMPLETE` is enabled.

Prediction names follow:

```text
output/<dataset>_<run-marker>_pos_predictions.csv
output/<dataset>_<run-marker>_noise_predictions.csv
```

## 5. Evaluate and Plot

Run:

```bash
python evaluate_picker_benchmark.py
python plot_picker_benchmark.py
```

Positive evaluation reports closest-pick comparisons, detection rate for
`|dt| <= 1 s`, residual mean/std, MAE/RMSE/OUT, threshold-dependent
precision/recall/F1, and PR/ROC-like curves.

Noise evaluation uses the number of indexed tested noise windows as the
denominator. It reports P&S false-positive ratio and the permissive P and S pick
ratios, both at emitted picks and over post-hoc probability thresholds.

Combined outputs are written under:

```text
output/picker_eval/<dataset>/<run-marker>/
output/picker_eval_noise/<dataset>/<run-marker>/
output/picker_eval_comparison/<comparison-marker>/
```

The combined figures include per-dataset P/S residual KDEs, the 2x4 positive
metric panel, and cross-dataset noise-stability metrics.
