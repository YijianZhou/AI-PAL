# Train Pickers

- `train_picker_local/`: shared sample cutting, one shared waveform Zarr plus
  frame/sample target arrays,
  continuous training via `3_train_eg.py`, and positive-window training via
  `3_train_pos_eg.py` for SAR, FT, PHN, and RUN.
- `train_picker_aws/`: SageMaker submission, container-entry, and monitoring
  workflows for sample cutting, Zarr conversion, and per-model GPU training.

Executable folders use two configuration layers:

- `config_ai_pal_eg.py` defines shared waveform preprocessing, sample
  construction, amplitude measurement, data readers, and PAL parameters.
- `config_sar_eg.py`, `config_ft_eg.py`, and the other model configs define
  only model structure, training-loop controls, and inference settings.

The `_eg` suffix marks case-specific executable examples. After renaming the
configs (for example, to `config_ai_pal_sc.py` and `config_sar_sc.py`), set the
single `CASE_CODE = "sc"` control in each local training launcher or
`CASE_CODE = "sc"` in each AWS submission or monitor script. Dataset names, prepared PAL labels, NPY/Zarr locations, checkpoint roots, and AWS run prefixes are then derived from it.
Source folders retain unsuffixed `config.py` model configs and inherit
`PAL_src/config_ai_pal.py`. Preprocessing modules use that picker-root
`config.py` directly; they do not maintain separate config copies. The distinct
`PAL_src/config_pal.py` file contains standalone defaults for rule-based PAL
runners and is not a picker model config.

`2_npy2zarr_eg.py` maps SAR/FT to integer frame targets and PHN/RUN to
Gaussian sample-resolution targets. It selects one builder per required family and
writes all enabled models into one shared Zarr root, so waveform data and each
target family are stored only once. Conversion is resumable: arrays with the
expected shape are preserved, while missing or shape-incompatible arrays are
written. Rerunning the converter can therefore add a newly required target
family without rewriting compatible waveform arrays.

Legacy SAC-to-Zarr helpers are archived under
References/backup/legacy_sac_preprocessing/; they are not staged by local or AWS
training workflows.

The packaged training workflows default to `CASE_CODE = "eg"`, where `eg` means "example".

## Local Training

Run from `train_picker_local/` after editing shared paths, `ENABLED_MODELS`, and
`gpu_idx`:

0. `python 0_analyze_phase_rarity_eg.py`
1. `python 1_cut_train-samples_eg.py`
2. `python 2_npy2zarr_eg.py`
3. `python 3_train_eg.py` for continuous-data picker training, or
   `python 3_train_pos_eg.py` for positive-window picker training.

Step 0 scores every station pick using the joint rarity of event magnitude/FMD,
hypocentral distance, and spatiotemporal seismicity rate. Epicentral distance
is calculated from event and station coordinates, including station elevation
in the hypocentral-distance term. It writes a PAL-compatible phase file with
tagged `num_aug`, `rarity`, `p_norm`, and distance fields, plus feature tables,
a summary, and the two diagnostic figures. Set `PHASE_FILE` in step 1 to this
rarity phase file.

`config_ai_pal_eg.py` exposes only the scientific controls: percentile
boundaries, augmentation values, maximum hypocentral distance, magnitude and
distance bin widths, and spatial/time density-bin sizes. Numerical smoothing,
floors, plotting resolution, and training-only bookkeeping are stable internal
defaults in `PAL_src/phase_rarity.py`. `positive_num_aug_mode = "phase"`
enables the new behavior.
Set it to `"fixed"` and point step 1 at the original PAL phase file to retain
the legacy global `num_aug` strategy. Validation samples are always cut once;
per-pick augmentation applies to training samples only. In rarity-aware mode,
negative sampling remains based on the original phase-pick count and each
station-date association ratio: `cut_neg_ratio = num_pos / num_unassociated`.
It is not increased or redistributed by per-pick positive augmentation.

The first step creates one shared NPY inventory. The second dispatches each
model's label converter, covering sample-segmentation and frame-label formats.
Each picker source (`picker_SAR`, `picker_FT`, `picker_PHN`, and `picker_RUN`)
also carries its own positive/negative NPY cutters and raw-shard helpers under
`preprocess/`. The combined workflow cuts the common inventory once through
SAR by default, but any picker folder can run the same cutting stage
independently using its own config and preprocessing modules.
The launchers stage `config_ai_pal_eg.py` for shared preprocessing and data
behavior, then stage one model-only `config_<model>_eg.py` per enabled picker.
Both training launchers process `ENABLED_MODELS` sequentially on one
`gpu_idx`. They derive each model source, config, and checkpoint subdirectory
from the model name while sharing one Zarr path and checkpoint root.

Standard training interprets `batch_size` as the number of positive windows;
negative windows are added according to the stored negative/positive ratio, up
to one negative per positive. Consequently, model computation can approach
twice the positive-only workload. The paired datasets preserve Zarr chunk
locality for both classes, and training prints the effective positive/negative
mix and chunk size at startup. `NUM_WORKERS` and `PREFETCH_FACTOR` remain
tunable in the launcher; excessive workers can reduce throughput on shared or
network storage.

`train_picker_local/input/eg_association_rates.csv` is a minimal single-file
example of the
station-date association counts written by PAL and consumed during negative-
sample preparation. Production inputs should concatenate all segmented rate
files for the same period as the phase file, retaining one CSV header.

For an older PAL run without association-rate output, edit
`pick_to_assoc_rate_eg.py` to point to the full-range pick and phase files. The
converter recognizes legacy PAL and newer AI pick formats, matches both P and S
times within the configured tolerance, and writes the consolidated training
CSV. A phase file is required because picks alone do not identify which
detections were associated.

The packaged training workflows default to `CASE_CODE = "eg"`, where `eg` means "example".

## Local Training Inputs

- `train_picker_local/input/eg_pal_hyp.pha`: one phase-label file covering the training period.
- `train_picker_local/input/eg_station.csv`: station coordinates and gains used
  to calculate epicentral/hypocentral distances in step 0.
- `train_picker_local/input/eg_association_rates.csv`: one station-date association-rate file covering
  the same period, in the format written by `PAL_src/association_runner.py`.

Concatenate segmented PAL association-rate outputs with one header. On Linux:

    awk 'FNR == 1 && NR != 1 {next} {print}' association_rate_*.csv > association_rates.csv

Plain `cat association_rate_*.csv` is also accepted by the reader, which skips
repeated header rows, but producing a single header is preferred. Set
`ASSOCIATION_RATE_FILE` in `train_picker_local/1_cut_train-samples_eg.py` to the resulting file.

Legacy runs that have a full-range pick file but no rate CSV can use
`train_picker_local/pick_to_assoc_rate_eg.py`; the matching full-range phase file is also required
to identify associated picks.

## AWS Training

The AWS workflow has four stages:

0. Prepare yearly PAL inputs and generate yearly rarity-aware phase files.
1. Cut shared positive and negative NPY samples by year.
2. Combine annual NPY inventories and build one shared waveform Zarr with the
   target families required by enabled models.
3. Train one GPU job for each enabled picker.

The current operational step is yearly NPY cutting for completed PAL years.

### AWS Layout

The reorganized AI-PAL checkout is expected at:

    /home/sagemaker-user/shared/software/AI-PAL

The completed PAL workspace may remain at its existing location:

    /home/sagemaker-user/shared/run_pal

Set `AI_PAL_ROOT` in each copied local or AWS launcher to the installed checkout.
It defaults to `~/shared/software/AI-PAL`; unfinished legacy PAL jobs are not
affected.

### Configuration Layers

The `train_picker_aws/` folder contains two kinds of example config:

- `config_ai_pal_eg.py` defines shared waveform preprocessing, sample layout,
  augmentation, reader functions, amplitude measurement, and PAL parameters.
- `config_sar_eg.py`, `config_ft_eg.py`, `config_phn_eg.py`, and
  `config_run_eg.py` define only model structure, training-loop controls, and
  inference settings.

The `_eg` suffix marks case-specific executable files. Each staged job receives
the shared case config as `PAL_src/config_ai_pal.py` and its model case config
as model-local `config.py`.

### Prepare Annual Labels

Edit the years tuple at the top of `0_prepare_cut_inputs_eg.py` when needed,
then run:

    cd ~/shared/software/AI-PAL/2_train_picker/train_picker_aws
    python 0_prepare_cut_inputs_eg.py

For each selected year, `0_prepare_cut_inputs_eg.py`:

- verifies every canonical daily phase file in the completed `<CASE_CODE>-assoc-YEAR`
  S3 output;
- concatenates those daily files chronologically into one PAL phase file;
- validates every daily association-rate CSV and combines their rows into one
  yearly CSV;
- copies the selected PAL station file from `~/shared/run_pal/input`.

The resulting structure under `train_picker_aws/` is:

    input/
      station_scedc_aws_selected_20200101_20260701_pal.csv
      2020/
        <CASE_CODE>_assoc_2020_pal.pha
        <CASE_CODE>_assoc_2020_association_rates.csv
      2021/
      2022/

Each yearly association-rate input contains one header and all station-date rows
for that year.

Waveforms are not downloaded here. Sample-cutting workers stream only requested
station-days from `s3://scedc-pds/continuous_waveforms/` using
`config_ai_pal_eg.py` and the shared `PAL_src/data_pipeline_training_aws.py` adapter.

After preparing the yearly files, run `0_analyze_phase_rarity_eg.py`. Its
`YEARS`, station path, and case code are user settings at the beginning of the
file. It creates `<case>_assoc_<year>_pal_rarity.pha` for the yearly cutting
job and stores the diagnostic CSV/JPG outputs under `output/`.

### Submit One Yearly Cutting Job

Edit `training_year` near the top of
`1_cut_train-samples_eg.py`. Keep the same `CASE_CODE` and `SAMPLE_RUN` for all
years that may be reused together. `SAMPLE_RUN` identifies the annual NPY
library; it is independent of the narrower `TRAINING_RUN` used for one Zarr
dataset and its model checkpoints.

    cd ~/shared/software/AI-PAL/2_train_picker/train_picker_aws
    AWS_DEFAULT_REGION=us-west-2 python 1_cut_train-samples_eg.py

Repeat for each completed year. Jobs may run concurrently only if the account
has enough capacity for the configured Processing instance; otherwise submit
sequentially.

Monitor every submitted year with:

    python processing_job/monitor_cut_samples_job.py

Annual artifacts are isolated at:

    s3://<default-bucket>/sagemaker/ai-pal/training/sc-2020-2025-v1/
      01_npy/2020/
      01_npy/2021/
      01_npy/2022/
      01_npy/2023/
      01_npy/2024/
      01_npy/2025/

Each completed annual directory contains positive and negative NPY shards,
four portable shard-index files, and `cut_samples_manifest.json`.

### Cutting Controls

The main controls are at the beginning of `1_cut_train-samples_eg.py`:

- `training_year`, `CASE_CODE`, and derived `SAMPLE_RUN`
- `instance_type` and `volume_size_gb`
- `num_workers` and `threads_per_worker`
- `shard_size`
- SCEDC bucket, region, access mode, and acceleration code

The Processing Job uses continuous S3 upload. A successful manifest is written
only after positive and negative cutting both finish.

### Build Zarr And Train

Do not start NPY-to-Zarr conversion until every year intended for the final
model is present under `01_npy/`. The converter consumes the complete annual
inventory so training and validation samples span the full selected period.

Annual NPY outputs are reusable. For example, one `SAMPLE_RUN` may contain
2020-2025, while separate `TRAINING_RUN` products select 2020-2021 and
2020-2025. Each product gets its own `02_zarr/` and `03_checkpoints/`; the
annual shards under the sample run are not copied again.

Model-to-target mapping is fixed: SAR/FT use integer frame labels, while
PHN/RUN use Gaussian sample-resolution soft labels. The enabled-model list is
reduced to the required target families, so waveform data and each target family
are written only once under `02_zarr/shared.zarr`.

Both positive and specially sampled negative waveforms and targets are stored;
negative targets are not generated as random noise during training. The AWS
Zarr job validates all required positive/negative arrays after writing. Each GPU
job validates its model-specific frame or sample targets again before starting
`train.py`, and records `training_mode: positive_negative` plus array shapes in
its manifest. Rebuild an older `shared.zarr` if it predates the stored negative
sample targets for PHN/RUN.

    python 2_npy2zarr_eg.py
    python processing_job/monitor_npy2zarr_job.py

After `shared.zarr` and all required target arrays finish:

    python 3_train_eg.py
    python processing_job/monitor_train_jobs.py

`2_npy2zarr_eg.py` and `3_train_eg.py` use the configs in
`train_picker_aws/`, not the local-training examples. Edit `enabled_models` and
the model registry in each submission script when selecting a subset.

## AWS Training Inputs

Run `0_prepare_cut_inputs_eg.py` in SageMaker JupyterLab. For every selected year it
creates one combined PAL phase file and one combined association-rate CSV, plus
the selected SCEDC PAL station file.

The phase and association-rate files are chronological concatenations of the
canonical daily PAL outputs. The association-rate CSV contains one header and
one row per station-date record. Waveforms are streamed directly from the SCEDC
public S3 continuous-waveform archive by the Processing Job and are not copied
here.
