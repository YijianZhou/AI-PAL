# AI-PAL AWS inference

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
