# AI-PAL Changelog

Major user-facing changes only. Implementation notes and the maintenance
procedure are in [CHANGELOG_INTERNAL.md](CHANGELOG_INTERNAL.md).

## Unreleased

Future changes will be recorded relative to v7.3.

## v7.3 - Release Preparation

Changes since v7.2. Prepared 2026-10-29; GitHub publication/tagging is performed separately.

- Improve PAL association with distinct-station counting, P/S residual checks,
  configurable origin-time grid search, adaptive spatial coverage, and faster
  candidate searches with reuse of failed-search results.
- Support precomputed travel-time archives built from JSON, combined Vp/Vs CSV,
  or paired CRE velocity models. Include example 1D model preparation and local
  table-building launchers for local and AWS workflows.
- Store travel-time arrays in float32 and share read-only tables across offline
  date workers. Existing float64 archives remain readable. Shared stores use
  executable `ASSOCIATION_TABLE_DIR`, never system temp, with checked writes
  and normal-run cleanup.
- Dynamically schedule unfinished PAL picking and association dates, retaining
  resume checks, waveform buffers, and persistent workers.
- Use `nan` for missing station gains; gain `1` is valid for calibrated data.
  Missing-gain stations remain usable for picking but not magnitude.
- Simplify default progress output while retaining detailed debug diagnostics.
  Multi-control launchers use stage-specific `XXX_NUM_WORKERS` names.
- Improve realtime CPU association/reassociation parallelism without duplicating
  GPU repickers; share read-only travel-time arrays across workers.
- Dynamically balance offline picking dates and postprocessing events across
  persistent model workers, with bounded CPU reassociation batches, atomic
  event checkpoints, and stable cross-process resume fingerprints.
- Export offline per-event and daily timing reports covering waveform discovery,
  reading, preprocessing, repicking, and checkpoint writes.
- Improve CPU/GPU picking concurrency and release unused GPU buffers/cache.
  Default offline waveform PNG export to off, retaining explicit opt-in.
- Add configurable magnitude-versus-station-participation QC. The default keeps
  the detection and marks suspicious magnitude as `nan`; removal and disabled
  modes are available with diagnostic CSVs.
- Align training defaults: positive batch size 128; Local negative batches 64
  (SAR/FT) and 16 (PHN/RUN), weight 1.0; CEED negative batches 16 and 4,
  respectively, weight 0.1 for all models. Existing checkpoints are unchanged.

### Migration Notes

- Replace old gain-1 placeholders with `nan` only where gains are unavailable.
- Runtime association uses homogeneous `velo_mod=[Vp, Vs]` or a saved
  `travel_time_table`; build layered-model tables before running association.
- Ensure `ASSOCIATION_TABLE_DIR` has enough disk space for shared tables.
  Per-worker search workspaces still require additional RAM.

## v7.2 - Release Preparation

Changes since v7.1. Prepared 2026-10-03; GitHub publication/tagging is performed separately.

### Detection and Quality Control

- Rename picker groups to Local and Global, identify Global models by dataset
  (for example SAR_CEED), and support both groups in continuous detection.
  Configure ensemble support independently for each group and for the total.
- Repick using shared random windows and distinct-window votes, then reassociate
  all QC-accepted Local/Global picks. Timing preference and pick quality are configurable.
- Require at least two Local+Global pairs and two quality-0 pairs per final event
  by default. Add ranked P-to-S overlap duplicate suppression, including linked
  descendants, with configurable inclusive thresholds and diagnostic outputs.
- Apply native waveform-glitch rejection before association. Keep realtime
  reference catalogs separate from AI-PAL-specific filtering.
- Improve calibration and PAL magnitude QC: retain arrival picks with unavailable
  gains, exclude uncalibrated amplitudes (marked by nan gains), and
  require at least three valid station magnitudes with standard deviation <=1.0.
  Failed event magnitudes use -1; other negative magnitudes remain supported.

### Training and Models

- Configure positive/negative batch sizes explicitly and weight negative-window
  loss independently. Validate full positive and negative sets separately,
  including negative validation when the negative training batch is zero.
- Improve integrated/annual Zarr loading with chunk-aligned positive sampling
  and reusable negative chunks; fix the FT validation-scheduler failure.
- Support optional CEED retraining with local negatives and synthetic Gaussian
  noise augmentation. Add resumable/multiprocess preparation and negative transfer,
  plus resumable local training-sample cutting.
- Refresh CEED and SoCal checkpoints and centralize packaged models under
  Pre-trained_models. Keep optional CEED preparation/training with the models;
  move case-specific benchmarking out of common production source.

### Workflows and Outputs

- Unify band/location selection and epoch-aware NET.STA.BAND.LOC station/gain
  inventories across local, AWS and realtime workflows; retain simplified inputs.
- Improve PAL date-block parallelism, buffered association, progress reporting,
  and resume behavior. Export integrated association rates for training handoff.
- Simplify offline launchers and visible phase/catalog outputs; keep intermediate
  status products under _internal.
- Publish compact final phase rows with instrument selectors, picker vote ratios,
  combined SNR fields, and separate detailed .picker_qc.csv companions.
  Event waveform PNG names include origin time and coordinates.
- Expand realtime monitoring of internal event stages, associated/repicked counts,
  and rejection diagnostics.

### Upgrade Notes

- Update workflow configs with the source; use Local/Global names and matching
  checkpoint/config pairs. Review explicit training batches and negative-loss weights.
- Use to_clean for raw-waveform cleaning (legacy to_prep remains supported).
  Review custom readers for compact phase rows and separate QC companions.
- Existing products are not repaired automatically. Reprocess affected data for
  calibration/magnitude, event QC, or duplicate-removal changes; do not combine
  older and newer processing policies in a comparison without a time cutoff.
- Negative magnitude -1 is reserved for failed PAL estimates. This PAL magnitude
  policy does not change MFT magnitude estimation.
- Future changes are recorded above under Unreleased, relative to v7.2. The
  source/config/documentation fingerprint is in RELEASE_BASELINE.json.

## v7.1 - Release Preparation

Compared with the supplied v7.0 release; reviewed 2026-09-17.

- Added resumable AWS continuous inference, with combined and separate
  picking, association, and postprocessing jobs.
- Added explicit realtime reference workflows: PHN-SB picks can be shared by
  subnet PAL and full-network GaMMA association. Preferred detection remains PAL.
- Reduced default SAR, FT, and RUN model sizes. Training now validates the
  full validation set every 5,000 steps and selects the best checkpoint by
  minimum validation loss; local training figures update automatically.
- Added P-only and S-only training labels using `-1` for the missing phase.
  Improved integrated/annual Zarr loading and CEED positive-picker training.
- Replaced repeated adjacent-day reads with rolling waveform buffers and
  shifted daily ownership. Reduced waveform copying and improved progress,
  monitoring plots, and restart handling.
- Revised phase outputs with configurable four-level location quality codes
  and per-picker window-vote ratios. Pick files no longer persist the rough
  station origin-time estimate; association computes it when needed.
- Added the station metadata, MassDownloader, reconciliation, daily merging,
  and continuity-check preprocessing workflow. PAL gains explicit geographic
  search bounds, trigger-count diagnostics, and direct final phase/catalog files.

### Upgrade Notes

- Use matching architecture configs and checkpoints; older SAR/FT/RUN
  checkpoints are not interchangeable with the reduced default architectures.
  Inference launchers require explicit checkpoint files. CEED positive-picker
  configs are named `config_<model>_pos_ceed.py`.
- Continuous inference now uses POS_NEG models only; positive-only models
  remain available for event repicking. Replace old reference-group settings
  with `reference_workflows` in realtime configs.
- Daily files now own `[D - buffer, D + 1 day - buffer)`. Do not mix old
  calendar-day files with shifted outputs without conversion or regeneration.
  Review phase-column parsers before consuming the revised quality fields.
- GaMMA references require `GMMA==1.2.12` and `scikit-learn==1.6.1`.
  See the [inference README](3_run_ai_pal/README.md) for current contracts.

## v7.0 - Comparison Baseline

The supplied release already includes the numbered package organization,
multi-picker ensembles, dual-group event postprocessing, and realtime
publication. Those are retained foundations, not new v7.1 features.
