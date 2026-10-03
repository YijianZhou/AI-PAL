# Local Picker Training

Use PAL detections, association-rate records, station metadata and cleaned local
waveforms to build a local training dataset. Run from this folder after editing
the script settings and config_ai_pal_eg.py:

1. python 0_analyze_phase_rarity_eg.py
2. python 1_cut_train-samples_eg.py
3. python 2_npy2zarr_eg.py
4. python 3_train_eg.py

The eg case is a template: replace paths and case-specific settings with your
real data. Shared waveform/scientific controls live in config_ai_pal_eg.py;
model architecture and training controls live in config_<model>_eg.py.

SAR/FT use [128, 32] positive/negative samples per training batch; PHN/RUN use
[128, 16]. Epoch length follows positives. Validation traverses the full positive
and negative splits independently every valid_step updates and at the final
update; class-balanced validation loss selects best.ckpt. [128, 0] explicitly
selects positive-only training/validation.

Local NPY indexes default to /data/bigdata/eg_train-samples_npy and the integrated
Zarr to /data/bigdata/eg_train-samples.zarr. Training outputs go to output/eg_ckpt.
See [the training overview](../README.md) for annual archive options.

### Resuming Sample Cutting

Rerun the same cutting launcher without deleting the NPY output folder. Both
cutters keep `.cut_resume` station-date completion records and validate shard
headers/sizes before skipping waveform reads. Interrupted or damaged station-date
outputs are regenerated; train/validation indexes are rebuilt from retained and
new shards. Unindexed leftover shards are ignored. Empty results are checkpointed.
The saved random seed preserves split assignments and per-item sampling when
worker counts change. Inputs, config sources, waveform root and shard size are
checked; changed inputs/settings require a new output root. Keep waveform contents
unchanged when resuming (the waveform tree is not hashed). Run only one cutting
launcher against an output root at a time.

Older completed stages can be adopted if their finished progress record and all
indexed shards validate. Since old outputs have no input fingerprint, this assumes
the same inputs/config as that original run and prints a warning. Older partial
stages without checkpoints are regenerated once. Copy `PAL_src/cut_resume.py`
along with the updated picker preprocessing scripts when updating the source.

## Global Pickers

Normally combine your trained Local models with the supplied Global CEED
models/configs from AI-PAL/Pre-trained_models/CEED/CEED_ckpt during inference.
You do not need to retrain CEED models or download CEED for this path.

Optional CEED preparation starts in
[preprocess_CEED](../../Pre-trained_models/CEED/README.md), followed by
[train_picker_CEED](../../Pre-trained_models/CEED/README.md).
That branch can copy negatives from a compatible integrated local Zarr.
It has its own numbered sequence and a single shared CEED waveform config.
Do not run launchers concurrently against one source installation: they stage
configuration files in the shared picker source directories.
