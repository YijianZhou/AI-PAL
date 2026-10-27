# AI-PAL AWS inference

## Optional Step 0: Local Travel-Time Preparation

Run `python 0_build_travel_time_table_eg.py` locally, not as an AWS job.
This uses the same CPU/NumPy builder as the local workflow and needs no AWS
credentials or GPU. Edit paths and parameters at the top; use `DRY_RUN = True`
first. Runtime depends on network size and grid resolution.

The bundled `input/velocity_model_1d.json` is an example, not a calibrated
SoCal model. Replace it before operational use. Combined Vp/Vs CSV and paired
P/S CRE files are also supported; examples are in `input`. Consult the installed
package's `PAL_src/VELOCITY_MODEL_FORMATS.md` for units, columns, depth-zero
elevation and bottom-depth settings. All preparation settings remain in step 0.

The default station inventory covers the full network; the grid inherits
`subnet_assoc_params['full']`. Output is `output/travel_time_table_eg` and
`input/travel_time_table.paltt`. Use a new output directory for each rebuild.
Ensure coverage of all required stations, grid cells and depths across subnets
and full-network reassociation; build separate tables if their grids differ.

Set `travel_time_table` to `'input/travel_time_table.paltt'` in the relevant
association config default/full/subnet entries to enable it. Existing `None`
defaults are unchanged. Submission uploads selected tables and rewrites paths
only in the staged config, including AI-PAL reassociation. Missing tables fail
before submission. Use literal path strings (relative to the workflow folder
or absolute), or `None`; no computed expressions. Local configs are untouched.

Offline postprocessing defaults to `POSTPROCESS_NUM_WORKERS = 4` model-owning
processes, each pulling same-day batches from a shared pending-event queue with its own
`ASSOCIATION_NUM_WORKERS = 4` CPU reassociation pool (up to 16 total). Both
settings are positive integers in step 3; association workers=1 disables the
inner pool. Repicking is serial per model worker with at most that many events
in flight; batches stop at day boundaries. `PREPROCESSING_NUM_WORKERS`
continues to control preprocessing threads per process. GPU/RAM requirements
scale with the process count. Daily and waveform upload callbacks execute in
the parent; the final requested-range outputs are assembled only after success.
Atomic event checkpoints under `_internal/postprocess_AI-PAL/events/` support
resume within unfinished days; keep/sync that folder to resume on a new host.
Daily merge and final plots/SAC run after all that day's events complete.

Station inputs must use the [SCEDC nine-column schema](../../STATION_FORMATS.md).
The launchers now reuse the full station file from
`../../1_run_pal/run_pal_aws/input/station_scedc_aws_selected_20200101_20260701_pal.csv`.
Their example dates and monitor run IDs are 20200704-20200707, within the
provided metadata period. The old `input/example_pal_format1.sta` is a local
five-column example, not a valid AWS input. Existing jobs/run IDs are not migrated.

This directory runs SCEDC inference in three resumable SageMaker Processing
jobs. Edit settings at the top of all three launchers, keeping `TIME_RANGE`
and `INFERENCE_RUN` consistent with `processing_job/monitor_staged_ai_pal_jobs.py`.
Check station files, checkpoint paths, GPU mapping, and instance type before
submission. Run the stages in order, waiting for each to finish:

```bash
python 1_run_ai_pal_pick_aws_eg.py
python 2_run_pal_assoc_aws_eg.py
python 3_run_ai_pal_repick_reassoc_aws_eg.py
python processing_job/monitor_staged_ai_pal_jobs.py
```

Stages `2` and `3` refuse submission until the preceding stage manifest is
present. Stage `1` picks one halo day before and after the target range;
association and postprocessing publish only target dates. In the `3`
launcher, set `SAVE_FILTERED_EVENT_WAVEFORMS = True` to write relocation-ready
SAC files under `3.1_phase_final_AI-PAL/event_waveforms/`. The SAC headers
retain origin, P, and S timing for the cross-correlation relocation workflow.

The launcher reads Local checkpoints from
`sagemaker/ai-pal/training/<TRAINING_RUN>/03_checkpoints/<MODEL>/`. Positive-only
CEED checkpoints are staged from `AI_PAL_ROOT/Pre-trained_models/CEED/CEED_ckpt`;
station files are staged from `input/`. Waveforms are read
directly from `s3://scedc-pds/continuous_waveforms/` with the channel epochs and
gains in the full PAL station file.

The CEED inputs are `ceed_sar_best.ckpt`, `ceed_ft_best.ckpt`,
`ceed_phn_best.ckpt`, and `ceed_run_best.ckpt` in that central CEED folder.
Temporary container bundles retain `workflow/input/CEED_ckpt/`; these are
job artifacts, not additional maintained package copies.
These use the same positive-picker model configs as realtime: SAR hidden size
128, FT width 256 with four heads and five layers, and RUN one block per stage.
Event postprocessing use these explicit checkpoint files.

Outputs are uploaded incrementally to
`s3://<default-bucket>/sagemaker/ai-pal/inference/<INFERENCE_RUN>/output/`.
There is deliberately no SageMaker managed output directory: direct uploads
avoid the Processing service's final artifact file-count limit. On resubmission,
the prior output is mounted and complete daily pick files are reused.

The default `ml.g4dn.12xlarge` supplies four T4 GPUs and mirrors the local
four-device model registry. A one-GPU instance is valid only after every model
is mapped to GPU 0 and the complete selected ensemble is confirmed to fit its
GPU memory.
Internal stage IDs and manifests retain their existing `2.1`/`2.2`/`2.3`
names for resume compatibility; only launcher numbering has changed.

# Offline Result Layout

The staged jobs publish `2.1_phase_init_AI-PAL/phase_<range>.dat`
and `catalog_<range>.dat` for initial detections, and equivalent range files in
`3.1_phase_final_AI-PAL` after postprocessing. Pick directories retain their
existing indexed names. Working association and postprocessing products are
under `_internal/`; monitors recognize both this layout and historical status
paths. Existing legacy association directories remain resumable without moving
them. Separate jobs may export different stages under different S3 prefixes;
the directory names inside each case output are consistent.

Continuous picking defaults to local SAR/PHN plus mixed-trained CEED SAR/PHN.
Set `picker_group_min_picker_support` (default `[0, 0, 1]`) to the minimum
Local, Global, and total votes for their combined ensemble; all must pass.
Pick-only jobs also require the CEED configs and centrally stored checkpoints.
Replace historical positive-only checkpoints with the newly mixed-trained CEED models.

PAL association accepts homogeneous `velo_mod=[Vp, Vs]` in km/s or a
precomputed `travel_time_table` (.paltt). Build layered-model travel times
separately; model-file paths and runtime solver options are no longer accepted.
