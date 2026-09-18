# AI-PAL AWS inference

This directory is the example SageMaker Processing workflow for continuous
SCEDC inference. It runs daily native AI-PAL picking, buffered hourly PAL
association, subnet merging, event repicking, and final reassociation in one
resumable GPU job.

Two execution layouts are provided:

- `1_run_ai_pal_pick_assoc_aws_eg.py` runs the complete workflow in one job.
- `2.1`, `2.2`, and `2.3` run picking, association, and event postprocessing
  as three independent jobs. Use the same `TIME_RANGE` and `INFERENCE_RUN` in
  all three launchers and in `monitor_staged_ai_pal_jobs.py`.

1. Edit the settings at the top of
   `1_run_ai_pal_pick_assoc_aws_eg.py`, especially `TIME_RANGE`, station files,
   `TRAINING_RUN`, `INFERENCE_RUN`, GPU mapping, and instance type.
2. Keep the matching values in
   `processing_job/monitor_ai_pal_job.py`.
3. Submit with `python 1_run_ai_pal_pick_assoc_aws_eg.py`.
4. Monitor with `python processing_job/monitor_ai_pal_job.py`.

For the staged workflow, submit in order:

```bash
python 2.1_run_ai_pal_pick_aws_eg.py
python 2.2_run_pal_assoc_aws_eg.py
python 2.3_run_ai_pal_repick_reassoc_aws_eg.py
python processing_job/monitor_staged_ai_pal_jobs.py
```

Stages `2.2` and `2.3` refuse submission until the preceding stage manifest is
present. Stage `2.1` picks one halo day before and after the target range;
association and postprocessing publish only target dates. In the `2.3`
launcher, set `SAVE_FILTERED_EVENT_WAVEFORMS = True` to write relocation-ready
SAC files under `3.1_phase_final_AI-PAL/event_waveforms/`. The SAC headers
retain origin, P, and S timing for the cross-correlation relocation workflow.

The launcher reads POS_NEG checkpoints from
`sagemaker/ai-pal/training/<TRAINING_RUN>/03_checkpoints/<MODEL>/`. Positive-only
CEED checkpoints and station files are staged from `input/`. Waveforms are read
directly from `s3://scedc-pds/continuous_waveforms/` with the channel epochs and
gains in the full PAL station file.

The CEED inputs are `ceed_pos_sar_best.ckpt`, `ceed_pos_ft_best.ckpt`,
`ceed_pos_phn_best.ckpt`, and `ceed_pos_run_best.ckpt` in `input/CEED_ckpt/`.
These use the same positive-picker model configs as realtime: SAR hidden size
128, FT width 256 with four heads and five layers, and RUN one block per stage.
Both combined and staged postprocessing use these explicit checkpoint files.

Outputs are uploaded incrementally to
`s3://<default-bucket>/sagemaker/ai-pal/inference/<INFERENCE_RUN>/output/`.
There is deliberately no SageMaker managed output directory: direct uploads
avoid the Processing service's final artifact file-count limit. On resubmission,
the prior output is mounted and complete daily pick files are reused.

The default `ml.g4dn.12xlarge` supplies four T4 GPUs and mirrors the local
four-device model registry. A one-GPU instance is valid only after every model
is mapped to GPU 0 and the complete selected ensemble is confirmed to fit its
GPU memory.
