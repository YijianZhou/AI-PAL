# AI-PAL Changelog

Major user-facing changes only. Implementation notes and the maintenance
procedure are in [CHANGELOG_INTERNAL.md](CHANGELOG_INTERNAL.md).

## Unreleased

- AWS workflows now include a local step-0 travel-time table builder (JSON,
  CSV or paired CRE); submissions automatically upload configured .paltt files.

- Build PAL travel-time tables from combined Vp/Vs CSV or paired P/S CRE,
  as well as JSON, with explicit preparation-only datum/bottom settings.

- Provide a configurable regional 1D velocity-model preprocessing example and
  document its JSON handoff to PAL/AI-PAL travel-time-table preparation.

- Hide detailed PAL association attempt/cache counters at normal console
  verbosity; retain them with console_verbosity="debug".

- Default offline event waveform plots to off; retain explicit opt-in and
  existing realtime defaults.

- Fix event-repick resume fingerprints changing between Python processes and
  avoid parsing input phase files for unchanged completed days.

- Run offline repicking with four persistent event workers by default, dynamic
  load balancing and atomic event-level resume; retain daily merging/products
  and station preprocessing concurrency.

- Default routine picking and association console progress to 60 seconds,
  including rapid date-completion updates; preserve immediate failure reports.

- Balance offline association dynamically across pending dates, retain worker
  caches, report active dates, and fail promptly on unexpected worker exits.

- Avoid repeating failed association searches after unrelated events are found,
  without restricting the spatial search or changing association thresholds.

- Reuse decoded station travel-time tables during association instead of
  repeatedly loading them; add flushed search diagnostics to daily logs.

- Added optional local `0_build_travel_time_table_eg.py` preparation with
  editable settings at the top and a shared `PAL_src` helper; no separate
  builder configuration file is needed.

- Use `nan` for unavailable station gains; gain `1` now correctly permits
  already-calibrated data to contribute to magnitude. Missing-gain stations
  remain usable for picking. Migrate older gain-1 placeholders before rerunning.

- Association now accepts only homogeneous `velo_mod=[Vp, Vs]` or a saved
  `travel_time_table` (.paltt). Build layered-model tables separately; runtime
  model-file inputs and `layered_model_options` have been removed from active
  workflows. Existing saved tables and homogeneous settings remain compatible.

- Load precomputed P/S times from one `.paltt` file using optional
  `travel_time_table`; skip unavailable station/cell pairs safely.

- Accept scalar `max_res=r` as shorthand for equal P/S residual limits `[r, r]`.

- PAL spatial association now checks both P and S arrivals. Configure homogeneous
  `velo_mod=[5.9, 3.5]` (km/s) and `max_res=[res_p, res_s]` (seconds);
  active workflow configs are migrated, retaining regional tolerances.

- Speed up PAL association with sliding-window station counts, local count
  updates, stable pick indices, and reused per-cluster calculations. Existing
  association settings and selection rules are preserved.

- PAL now counts distinct stations by default (`unique_stations=True`) while
  retaining alternative P-S pairs for spatial selection. Failed neighborhoods
  yield to alternative centers; retries no longer discard their candidates.
  Fix adaptive grid coverage and preserve internal location precision. Catalog
  coordinates use four decimal places for lat/lon and one for depth.

- Configurable final AI-PAL station-participation QC flags unusually large
  magnitudes supported by few stations within the Local+Global pick radius.
  Default action preserves detections but marks magnitude `nan`; removal and
  disabled modes are available, with per-event diagnostic CSVs.

- PAL association now searches up to ten origin times at 0.1-second spacing
  near the cluster median, bounded by its 5th-95th percentiles. Solutions maximize
  associated count, then minimize median absolute P residual. Temporal step,
  maximum span, and point count are configurable per subnet (defaults: 0.1 s,
  1.0 s, and 10 points).


- Faster offline picking resume: skipped days no longer reload waveform tails;
  only the context needed by the next unfinished day is loaded.

- Offline GPU picking now releases unused CUDA allocator cache after each
  successful station/device group. This reduces idle reservation, not the
  required peak inference memory; CPU picking and checkpoints are unchanged.

- Offline CPU pickers now run concurrently by model within each date-block
  process, while GPU inference remains serialized per device.
- Release finished GPU waveform copies without waiting for CPU pickers, and
  avoid unused SAR attention weights. Existing checkpoints remain compatible.
- Local AI-PAL pick-only examples now default to four date-block processes;
  preprocessing concurrency remains four workers per process.
- Local training defaults now use 64 negative windows for SAR/FT and 16 for
  PHN/RUN, with negative loss weight 1.0. CEED training uses 16 and 4 respectively,
  with negative loss weight 0.1 for every model. Positive batch size remains 128.
  Align these defaults across active workflow, benchmark, and pretrained-model
  config copies; existing checkpoint weights and inference settings are unchanged.

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
