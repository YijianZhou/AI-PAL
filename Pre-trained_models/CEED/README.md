# CEED Global Pickers

Step `train_picker/2.2_copy_neg_zarr_ceed.py` defaults to `NUM_WORKERS = 4`
for parallel annual negative copying; set 1 for serial execution. Workers own
disjoint full storage chunks/shards and only the parent updates checkpoints.
Existing interrupted transfers resume with either worker count. `BATCH_ROWS`
is aligned to storage chunk size; memory scales with worker count. More workers
are not necessarily faster when disk bandwidth is saturated. Do not train or
modify the stores during transfer. Use screen/tmux for long remote runs and
ensure old workers have stopped before restarting after a forced parent kill.

Setting a zero negative training batch does not disable negative validation.
The mixed Zarr must still contain a nonempty negative validation set; all four
trainers evaluate both classes and report negative accuracy at every validation.

This folder groups the supplied Global CEED models, optional preparation and
retraining workflows, and original dataset documentation. Most users can reuse
the models without downloading CEED or retraining.

## Folder Layout

- `CEED_ckpt/`: supplied checkpoints and matching inference model configs.
- `preprocess/`: raw download, phase extraction, optional fixed-window shards,
  and shared `config_ai_pal_ceed.py`.
- `train_picker/`: rarity analysis, augmented training shards, mixed Zarr and training.
- `CEED_doc/`: original dataset loader and usage example.
- `helpers/`: CEED HDF5/window preparation, negative-sample transfer, and shared
  config loading. These dataset-specific modules are not part of `PAL_src`.
- `tests/`: CEED preparation, transfer, workflow and checkpoint-distribution tests.

Deploy this CEED directory as a unit: preparation/training launchers locate
`helpers/` beside their own folders. The SoCal annual-negative transfer launcher
uses `AI_PAL_ROOT/Pre-trained_models/CEED/helpers`. General training/conversion
and production inference utilities remain in the installed AI-PAL source tree.

Continuous waveform and station-file preparation remain in
[AI-PAL/preprocess](../../preprocess/README.md). Local training remains in
[train_picker_local](../../2_train_picker/train_picker_local/README.md).

## Reuse Models

The supplied files are `ceed_sar_best.ckpt`, `ceed_ft_best.ckpt`,
`ceed_phn_best.ckpt`, and `ceed_run_best.ckpt`. All CEED model config filenames
use `config_<model>_global_ceed.py`, including the optional training workflow.
The shared waveform config remains `config_ai_pal_ceed.py`.

Use checkpoints and matching configs from [CEED_ckpt](CEED_ckpt) alongside your
Local pickers. Keep model architecture and waveform settings consistent with
the checkpoint. The scripts below describe the training workflow; placeholder
paths are not an exact provenance record for an individual supplied checkpoint.

## Optional Preparation

Run from `preprocess/` after editing `AI_PAL_ROOT` and dataset paths:

1. `python 0_download_ceed_raw.py`: CEED metadata and NC/SC raw HDF5 archives.
2. `python 1_extract_ceed_phase.py`: `output/ceed_phase.pha` and station counts.
3. Optional: `python 2_build_ceed_fixed_window_npy.py`: fixed-window inspection
   shards. These are not the augmented training dataset used by the converter.

Download and phase extraction do not filter waveforms. Existing fixed-window
output is not cleared by default. Set `P_TARGET_SEC` below `win_len`; both
arrivals must fit inside the window. `ceed_fixed_window_utils.py` is a helper,
not an execution step. SeisBench utilities remain in
`SoCal_workdir/benchmark_picker/preprocess_SeisBench`.

Phase extraction supports per-HDF5 multiprocessing (`NUM_WORKERS=4`) and resume
(`RESUME=True`). `PART_DIR`, defaulting to `ceed_phase_parts` beside `PHASE_OUT`,
retains phase/station-count parts and completion markers. Rerunning skips parts
whose source path, size, modification time and extraction version still match.
Interrupted files restart individually, not from their last event. The final
outputs are assembled in deterministic input-file order after every selected
file succeeds; failures leave completed parts reusable and are listed in
`PART_DIR/failures.json`. Existing monolithic outputs alone cannot be resumed.
Keep the parts for future reruns; use `RESUME=False` to force regeneration.
Workers open HDF5 independently; reduce their count if RAM or storage is limited.
An OS lock prevents concurrent extraction jobs using the same part directory.

## Optional Training

This is a dataset-specific reproducibility workdir for the supplied models,
not a required part of the common continuous-data workflow. Its cutting recipe
matches `SoCal_workdir/train_picker_CEED`; paths and negative-source layouts
remain case-specific. Both workflows separate step 2.1 (positive conversion)
and step 2.2 (annual local-negative transfer).

Continue from `train_picker/`:

1. `python 0_analyze_phase_rarity_ceed.py`: station-level split and rarity tags.
2. `python 1_cut_train-samples_ceed.py`: augmented train/validation NPY shards
   directly from raw HDF5 and the rarity annotations.
3. `python 2.1_build_pos_zarr.py`: build CEED positive Zarr arrays and targets.
4. `python 2.2_copy_neg_zarr_ceed.py`: copy annual local negatives into that Zarr.
5. `python 3_train_ceed.py`: train SAR, FT, PHN and RUN sequentially.

`/data/Example_data/...` and the `eg-rarity-v1` source name are fictional placeholders.
Supply real paths, then set `DATASET_SETTINGS_CONFIRMED = True` for conversion
and training. Do not run launchers concurrently against one installed source:
they stage configuration files in the shared source folders.

## Shared Configuration

Both folders use [preprocess/config_ai_pal_ceed.py](preprocess/config_ai_pal_ceed.py).
Window length, sample rate, frequency band, filter switch, taper length, rarity
settings and validation fraction are configured there. Conversion/training stage
the same file into the installed source. Raw HDF5 still needs format-specific
cleaning; `to_filter` independently controls bandpass filtering.

### Offline Noise Augmentation

Training shards use the per-row `num_aug` from the rarity phase file. The first
copy has no added noise; subsequent copies receive independent three-component
Gaussian noise. Validation always has one copy and no added noise, even if its
phase row requests more copies. As in Local cutting, every copy has a randomly
positioned window containing both arrivals with the configured buffer.

Noise is generated on a padded record, preprocessed with the configured bandpass
when enabled, cropped and peak-normalized using `global_max_norm`. One random
multiplier `alpha ~ Uniform(0, max_noise)` is drawn per augmented copy. Each
component receives `alpha * peak(abs(signal between P and S)) * normalized_noise`.
The mixed window is demeaned and normalized again. Thus the amplitude rule is
the Local rule, with synthetic noise replacing same-station/day noise; it is not
a target SNR or a Gaussian standard deviation in physical units. `max_noise=0`
disables added noise while retaining time shifts. Existing background noise in
the CEED recording is retained. No online noise probability/type settings exist.

The cutter summary records the recipe, `max_noise`, normalization mode and counts
of copies with/without added noise. Its seed plus sorted HDF5 group index makes
augmentation reproducible for the same inputs, independent of worker scheduling.

To rebuild an existing dataset, reuse the rarity phase file and restart at step
1; no raw download or rarity rerun is required unless those inputs changed.
Choose a new empty NPY directory and a new Zarr path, use these same paths in
conversion/negative transfer/training, then copy the local negatives again.
Existing shards and Zarr stores are not overwritten. Old checkpoints are not
modified. Noise is baked into positive NPY shards and copied unchanged by Zarr
conversion; negative data, labels and full validation behavior are unchanged.

Model settings are in `train_picker/config_<model>_global_ceed.py`. Batch sizes are
`[128, 16]` for SAR/FT with `negative_loss_weight = 0.5`, and `[128, 4]`
for PHN/RUN with `negative_loss_weight = 0.1`. The multiplier applies only
to negative training windows; all data points inside positive windows retain
weight 1. The training loss is `(Bp*Lp + w*Bn*Ln)/(Bp + w*Bn)`;
validation remains equally weighted between the full positive/negative sets.
These are retraining settings, not a change to supplied checkpoint weights.
Nonzero negative batches
require train and validation negative arrays. Validation traverses the complete
positive and negative splits independently; class-balanced loss selects best.ckpt.

## Data Handoff

Negative training uses the shared chunk-buffered sampler: each worker consumes
its disjoint negative partition before reshuffling, retaining its position across
positive epochs. Only one decoded chunk of waveforms/targets is cached per worker.
Batch sizes, positive sampling and full class-wise validation are unchanged.
Restarted jobs begin a new traversal; sampler positions are not checkpointed.

Training rarity reads `../preprocess/output/ceed_phase.pha`, then writes
`output/ceed_phase_train_augmented.pha`. The cutter reads that file and writes
`/data/Example_data/ceed_train_npy/{train_pos,valid_pos}.npy` shard indexes.
Conversion and training share `/data/Example_data/ceed_train-samples.zarr`.

Shards have shape `(samples, 3, round(win_len * samp_rate) + 2)`: relative P/S
times occupy the first two values per channel, followed by waveform samples.
Fixed-window shards are independently normalized for inspection; training shards
go through the training normalization and target-conversion pipeline.

In step 2.2, replace `LOCAL_ZARR=/data/Example_data/eg-rarity-v1` with your own
annual archive directory containing `<year>.zarr`, and set `TRAINING_YEARS`.
Keep step 2.1 `OUT_ZARR`, step 2.2 `CEED_ZARR`, and step 3 `ZARR_PATH` identical.
Waveform shape, sample rate, filtering and
target families must agree. Train/validation negatives retain their splits;
the source is not modified. Copy checks validate structure, not filter semantics.
Step 2.2 uses the same resumable annual transfer as the SoCal workdir.

Step 2.1 refuses existing output stores; step 2.2 requires the store it built
and preserves its positive arrays. Do not train from incomplete transfers.
Keep resulting checkpoints paired with the model and waveform configs.

## Original Dataset Documentation

The original dataset card declares license: MIT. Its description,
acknowledgments and HDF5 example are preserved below; this does not relicense
the AI-PAL package.


## CEED: *C*alifornia *E*arthquake *E*vent *D*ataset for Machine Learning and Cloud Computing

The California Earthquake Event Dataset (CEED) is a dataset of earthquake waveforms and metadata for machine learning and cloud computing. 

Detailed statistics about the dataset are available in this [arXiv paper](https://arxiv.org/abs/2502.11500).

### Acknowledgments

The seismic data used in this study were collected by (1) the Berkeley Digital Seismic Network (BDSN, doi:10.7932/BDSN) and the USGS Northern California Seismic Network (NCSN, doi:10.7914/SN/NC); and (2) the Southern California Seismic Network (SCSN, doi:10.7914/SN/CI).
The original waveform data, metadata, and data products for this study were accessed through the Northern California Earthquake Data Center (doi:10.7932/NCEDC) and the Southern California Earthquake Center (doi:10.7909/C3WD3xH1).
Please include acknowledgments and citations of the original data providers when using this dataset.

The dataset structure is shown below, and you can find more information about the format at [AI4EPS](https://ai4eps.github.io/homepage/ml4earth/seismic_event_format1/)

```
 Group: / len:60424
  |- Group: /ci38457511 len:35
  |  |-* begin_time = 2019-07-06T03:19:23.668000
  |  |-* depth_km = 8.0
  |  |-* end_time = 2019-07-06T03:21:23.668000
  |  |-* event_id = ci38457511
  |  |-* event_time = 2019-07-06T03:19:53.040000
  |  |-* event_time_index = 2937
  |  |-* latitude = 35.7695
  |  |-* longitude = -117.5993
  |  |-* magnitude = 7.1
  |  |-* magnitude_type = w
  |  |-* nt = 12000
  |  |-* nx = 35
  |  |-* sampling_rate = 100
  |  |-* source = SC
  |  |- Dataset: /ci38457511/CI.CCC..HH (shape:(3, 12000))
  |  |  |- (dtype=float32)
  |  |  |  |-* azimuth = 141.849479
  |  |  |  |-* back_azimuth = 321.986302
  |  |  |  |-* component = ENZ
  |  |  |  |-* depth_km = -0.67
  |  |  |  |-* distance_km = 34.471389
  |  |  |  |-* dt_s = 0.01
  |  |  |  |-* elevation_m = 670.0
  |  |  |  |-* event_id = ['ci38457511' 'ci38457511' 'ci37260300']
  |  |  |  |-* instrument = HH
  |  |  |  |-* latitude = 35.52495
  |  |  |  |-* local_depth_m = 0.0
  |  |  |  |-* location = 
  |  |  |  |-* longitude = -117.36453
  |  |  |  |-* network = CI
  |  |  |  |-* p_phase_index = 3575
  |  |  |  |-* p_phase_polarity = U
  |  |  |  |-* p_phase_score = 0.8
  |  |  |  |-* p_phase_status = manual
  |  |  |  |-* p_phase_time = 2019-07-06T03:19:59.422000
  |  |  |  |-* phase_index = [ 3575  4184 11826]
  |  |  |  |-* phase_picking_channel = ['HHZ' 'HNN' 'HHZ']
  |  |  |  |-* phase_polarity = ['U' 'N' 'N']
  |  |  |  |-* phase_remark = ['i' 'e' 'e']
  |  |  |  |-* phase_score = [0.8 0.5 0.5]
  |  |  |  |-* phase_status = manual
  |  |  |  |-* phase_time = ['2019-07-06T03:19:59.422000' '2019-07-06T03:20:05.509000' '2019-07-06T03:21:21.928000']
  |  |  |  |-* phase_type = ['P' 'S' 'P']
  |  |  |  |-* s_phase_index = 4184
  |  |  |  |-* s_phase_polarity = N
  |  |  |  |-* s_phase_score = 0.5
  |  |  |  |-* s_phase_status = manual
  |  |  |  |-* s_phase_time = 2019-07-06T03:20:05.509000
  |  |  |  |-* snr = [ 637.9865898   286.9100766  1433.04052911]
  |  |  |  |-* station = CCC
  |  |  |  |-* unit = 1e-6m/s
  |  |- Dataset: /ci38457511/CI.CCC..HN (shape:(3, 12000))
  |  |  |- (dtype=float32)
  |  |  |  |-* azimuth = 141.849479
  |  |  |  |-* back_azimuth = 321.986302
  |  |  |  |-* component = ENZ
  |  |  |  |-* depth_km = -0.67
  |  |  |  |-* distance_km = 34.471389
  |  |  |  |-* dt_s = 0.01
  |  |  |  |-* elevation_m = 670.0
  |  |  |  |-* event_id = ['ci38457511' 'ci38457511' 'ci37260300']
  ......
  ```

## Getting Started

### Requirements
- datasets
- h5py
- fsspec
- pytorch

### Usage
Import the necessary packages:
```python
import h5py
import numpy as np
import torch
from datasets import load_dataset
```
We have 6 configurations for the dataset: 
- "station"
- "event"
- "station_train"
- "event_train"
- "station_test"
- "event_test"

"station" yields station-based samples one by one, while "event" yields event-based samples one by one. The configurations with no suffix are the full dataset, while the configurations with suffix "_train" and "_test" only have corresponding split of the full dataset. Train split contains data from 1970 to 2019, while test split contains data in 2020.

The sample of `station` is a dictionary with the following keys:
- `data`: the waveform with shape `(3, nt)`, the default time length is 8192
- `begin_time`: the begin time of the waveform data
- `end_time`: the end time of the waveform data
- `phase_time`: the phase arrival time
- `phase_index`: the time point index of the phase arrival time
- `phase_type`: the phase type
- `phase_polarity`: the phase polarity in ('U', 'D', 'N')
- `event_time`: the event time
- `event_time_index`: the time point index of the event time
- `event_location`: the event location with shape `(3,)`, including latitude, longitude, depth
- `station_location`: the station location with shape `(3,)`, including latitude, longitude and depth

The sample of `event` is a dictionary with the following keys:
- `data`: the waveform with shape `(n_station, 3, nt)`, the default time length is 8192
- `begin_time`: the begin time of the waveform data
- `end_time`: the end time of the waveform data
- `phase_time`: the phase arrival time with shape `(n_station,)`
- `phase_index`: the time point index of the phase arrival time with shape `(n_station,)`
- `phase_type`: the phase type with shape `(n_station,)`
- `phase_polarity`: the phase polarity in ('U', 'D', 'N') with shape `(n_station,)`
- `event_time`: the event time
- `event_time_index`: the time point index of the event time
- `event_location`: the space-time coordinates of the event with shape `(n_staion, 3)`
- `station_location`: the space coordinates of the station with shape `(n_station, 3)`, including latitude, longitude and depth

The default configuration is `station_test`. You can specify the configuration by argument `name`. For example:
```python
# load dataset
# ATTENTION: Streaming(Iterable Dataset) is complex to support because of the feature of HDF5
# So we recommend to directly load the dataset and convert it into iterable later
# The dataset is very large, so you need to wait for some time at the first time

# to load "station_test" with test split
ceed = load_dataset("AI4EPS/CEED", split="test")
# or
ceed = load_dataset("AI4EPS/CEED", name="station_test", split="test")

# to load "event" with train split
ceed = load_dataset("AI4EPS/CEED", name="event", split="train")
```

#### Example loading the dataset
```python
ceed = load_dataset("AI4EPS/CEED", name="station_test", split="test")

# print the first sample of the iterable dataset
for example in ceed:
    print("\nIterable test\n")
    print(example.keys())
    for key in example.keys():
        if key == "data":
            print(key, np.array(example[key]).shape)
        else:
            print(key, example[key])
    break

# %%
ceed = ceed.with_format("torch")
dataloader = DataLoader(ceed, batch_size=8, num_workers=0, collate_fn=lambda x: x)

for batch in dataloader:
    print("\nDataloader test\n")
    print(f"Batch size: {len(batch)}")
    print(batch[0].keys())
    for key in batch[0].keys():
        if key == "data":
            print(key, np.array(batch[0][key]).shape)
        else:
            print(key, batch[0][key])
    break
```

<!-- #### Extension

If you want to introduce new features in to labels, we recommend to make a copy of `CEED.py` and modify the `_generate_examples` method. Check [AI4EPS/EQNet](https://github.com/AI4EPS/EQNet/blob/master/eqnet/data/quakeflow_nc.py) for an example. To load the dataset with your modified script, specify the path to the script in `load_dataset` function:
```python
ceed = load_dataset("path/to/your/CEED.py", name="station_test", split="test", trust_remote_code=True)
```
 -->
