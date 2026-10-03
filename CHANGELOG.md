# AI-PAL Changelog

Major user-facing changes only. Implementation notes and the maintenance
procedure are in [CHANGELOG_INTERNAL.md](CHANGELOG_INTERNAL.md).

## Unreleased

- Require at least two Local+Global station P/S pairs per final AI-PAL event by default.

- Make second-class duplicate thresholds inclusive: OT gap <= tolerance,
  pick overlap >= threshold, and failed-pick fraction >= threshold.

- Training NPY cutting now resumes completed station-dates, regenerates incomplete
  shards, and preserves randomized splits across restarts.

- Fix negative training-sample cutting for station-days with no catalog picks.

- 2026-10-01: PAL range exports include one integrated association-rate CSV
  beside phase/catalog outputs, ready for training sample preparation.

- Add configurable second-class duplicate suppression for final AI-PAL events:
  ranked P-to-S overlap links, including descendants of discarded events, with
  per-link diagnostic CSVs. Available in local, AWS, and realtime workflows.

- Native realtime pickers now reject waveform glitches before initial
  association, consistent with local/AWS picking. Reference PHN-SB catalogs
  no longer receive PAL post-association glitch filtering. Initial-QC rejection
  bars are removed from monitoring; amplitude/magnitude measurement remains.

- 2026-09-30: Final event repicking can prefer Global (default) or Local
  timing using `repick_timing_preference` in the shared configuration.

### 2026-09-30: Unit-Gain Placeholder

- Treat gain 1.0 as missing calibration in local/AWS PAL and AI-PAL realtime
  processing, including legacy gain layouts. Retain waveform picking but output
  NaN station amplitude and exclude it from magnitude calculation if a selected
  component is uncalibrated. Existing amplitudes require reprocessing.
- AI-PAL repick version is advanced to invalidate previous repick completion
  records. Regression tests cover unit gain on all or one component and both
  AI-PAL/PALM calibration implementations.

### 2026-09-30: PAL Magnitude QC

- Require three distinct valid station magnitude estimates and population std
  <= 1.0, configurable with `mag_min_stations` and `mag_max_std`.
- Replace unconditional worst-station removal with a spread check followed by
  the median. Failed magnitude is -1 without discarding arrival picks or events;
  missing-gain amplitudes remain NaN. Magnitude merging excludes the -1 sentinel
  and retains other negative values. This supersedes the earlier NaN event-mag
  convention for PAL; MFT magnitude estimation is unchanged.
- Shared helper and parameter propagation cover local/AWS PAL and AI-PAL
  realtime/reassociation. Repick completion checks include magnitude thresholds.
  Tests cover the reported M6.56 case, missing amplitudes, distinct stations,
  negative magnitudes, threshold boundaries and AI-PAL/PALM source parity.

- Final AI-PAL events additionally require two quality-0 station P/S pairs by
  default (`final_event_min_quality0_picks`); zero disables this requirement.

- Final AI-PAL events now require at least one Local+Global station P/S pair
  by default across local, AWS and realtime; the minimum is configurable.

- 2026-09-29: Centralized packaged inference checkpoints in
  `Pre-trained_models`; removed verified duplicate input copies.

- CEED phase extraction now supports per-HDF5 multiprocessing and resumable
  completed-file outputs in both the packaged and SoCal workflows.

- Packaged CEED retraining now matches the workdir's separate positive-Zarr
  build (2.1) and resumable annual-negative transfer (2.2), using example paths.

- CEED training-shard preparation now adds randomly scaled synthetic Gaussian
  noise to copies after the first, following Local augmentation semantics.
  Validation remains unaugmented by noise; existing datasets must be rebuilt.

- Repicking now merges P/S pairs using distinct-window support and the shared
  multi-picker clustering algorithm, independent of predicted arrival ranking.
  Random windows retain the full phase buffer. Final duplicate merging matches
  NET.STA while preserving instrument selectors and recalculating support/quality.

- Added configurable negative-window training loss weighting for all four
  pickers, with separate group-loss diagnostics and unchanged full validation.
  Updated CEED retraining and SoCal case batch/weight recipes.

- Final waveform-measured phase and QC rows retain the selected channel band
  and location (`NET.STA.BAND.LOC`); locator readers accept these identifiers.

- Repick quality 0 now also includes three strong pickers within either Local
  or Global, configurable through `pick_quality_code0_min_pickers`.

- Event reassociation now uses all QC-accepted Local and Global repicks,
  including single-group picks. Removed the later residual-matched
  supplementation pass; existing QC and quality codes remain unchanged.

- All four picker trainers now evaluate the full negative validation set even
  when the negative training batch is zero, including metrics and balanced
  validation loss used for best-checkpoint selection.

- Default training batches are now [128, 32] for SAR/FT and [128, 8] for
  PHN/RUN. The dedicated CEED training recipe retains its smaller 8/4 negative batches.

- Picker architecture is inferred from the name prefix (e.g. SAR_CEED -> SAR).
  Removed redundant model keys from local/AWS/realtime picker settings.

- Fixed event repicking for dataset-qualified model names such as SAR_CEED
  and FT_CEED: architecture-specific input framing and inference are now
  selected using the configured model type, not its display name.

- Missing instrument gains no longer remove arrival-time picks. Same-band
  location/epoch fallback is allowed; no cross-band gain borrowing. Uncalibrated
  amplitudes are nan and excluded from event magnitude.

- 2026-09-25: realtime processing skips stations with missing instrument/epoch
  gains with a warning instead of aborting the whole segment; calibration
  remains strict and uncalibrated picks are not published.

- 2026-09-25: refreshed bundled SoCal 2020-2025 picker checkpoints and realtime
  example copies; training batches are [128, 32] for SAR/FT and [128, 16] for PHN/RUN.

- 2026-09-24: PAL amplitude QC no longer crashes on incomplete component
  windows. Unavailable checks are recorded as `nan` and bypassed; available
  failing checks still reject picks.

- 2026-09-24: Retired the combined offline AI-PAL launcher. Local and AWS
  inference now use stages 1 (pick), 2 (associate), and 3 (repick/reassociate).
  Existing output paths and resume manifests are unchanged; realtime is unchanged.

- AI picking batch size is configured only in the workflow config; the local
  example defaults to 256 without a pick-only launcher override.

- Local AI pick-only step 2.1 supports spawned date blocks (default 2), retaining
  per-process preprocessing concurrency (4). All picker device defaults use GPU
  0, with batch size 256, private configs, per-block logs and GPU peak reporting.

- Move CEED-specific preprocessing/negative-transfer helpers and tests into
  `Pre-trained_models/CEED`, keeping dataset-specific workflows out of `PAL_src`.

- Clarify window-inference modules as `picker_window.py`; NPY-shard benchmark
  runners/adapters now live in the SoCal benchmark workdir instead of the
  source picker folders. Production repicking uses the shared window engines.


- Refresh bundled CEED inference checkpoints across local, AWS, and realtime
  workflows. Checkpoints use `ceed_<model>_best.ckpt`; CEED picker configs use
  `config_<model>_global_ceed.py` consistently, including optional training.

- Local picking exposes `threads_per_worker` for native numerical libraries,
  matching the AWS control; example launchers default to 2.

- Local PAL picking now uses independent date-block processes instead of
  station threads. Worker count controls date blocks; daily outputs and resume
  checks are unchanged, with per-block logs and aggregate console progress.

- Simplified local PAL entry points to `1_run_pal_pick_eg.py` and
  `2_run_pal_assoc_eg.py`; removed the redundant combined launcher.

- Local picking resumes without reading waveform tails for every skipped day;
  previous-day context is loaded only when picking actually resumes.

- Fixed daily trigger-count ownership at buffered boundaries: pre-QC candidates
  now use refined P time like accepted picks, avoiding false count failures.

- Local picking retains waveforms with missing gain locations/epochs, using
  warned same-band gain fallback or uncalibrated counts as a last resort.

- Local PAL picking now prints live day/station progress and a 30-second
  heartbeat while retaining detailed output in a line-buffered log.

- Daily PAL association now shares buffered picks across all subnets inside
  parallel contiguous date blocks, eliminating per-subnet pick-file rereads.

- PAL configs explicitly use a 30 s association buffer. Local PAL examples
  now enable cross-day association and pick halos, with bounded per-worker
  daily pick caching. Association resumes detect buffer-policy changes.


- Align positive training batches to each annual Zarr store's physical chunks
  for all four pickers, avoiding unnecessary cross-chunk reads.
- Negative training now reuses decoded Zarr chunks across minibatches, with
  disjoint worker partitions and traversal continuing across positive epochs.
  Positive sampling, batch ratios and validation are unchanged.
- Separated optional CEED preparation/retraining from local picker training;
  numbered each workflow independently and shared CEED waveform settings across
  preparation and training. Local users can reuse the supplied Global models.
- Renamed raw waveform cleaning control to `to_clean`, distinct from filtering;
  legacy `to_prep` configs and reader keyword arguments remain supported.
- Final pick quality counts votes at the configured strong-vote threshold
  inclusively (`>=`); a ratio of exactly 0.5 now qualifies by default.
- Default picker trigger thresholds to 0.3 across Local/Global models; move
  final quality settings into post-processing and standardize waveform option
  names to singular (`save_filtered_event_waveform`).
- Aligned the SoCal realtime case wrapper with Local/Global continuous picking,
  dataset-qualified CEED models, and current source configuration/CLI names.
- Compact final phase rows to 13 columns: picker identities are included in
  vote ratios, SNR components share one field, and probability spreads move
  to the QC companion. Shared readers retain legacy-format support.
- Fixed FT training failing after its first update because the validation
  scheduler referenced an undefined batch-count variable.
- Separated CEED preprocessing from case-specific SeisBench benchmark utilities;
  the CEED fixed-window builder no longer depends on the SeisBench builder.
- All four trainers validate the entire positive and negative splits separately,
  select best checkpoints using class-balanced validation loss, report both
  class losses, and always validate at the final training step.
- Training mode now follows `batch_size` alone: `[128, 0]` trains and validates
  using positives only. Removed the separate positive-only CLI switch;
  converters select source classes with `--sample_types` instead.
- The historical benchmark `train_pos/3_train_pos_pickers.py` now trains on
  CEED positives plus imported local negatives, with reduced negative batches.
- Annual CEED negative transfers now checkpoint and resume interrupted copies;
  legacy interrupted transfers have an explicit, positive-preserving restart.
- Continuous picker support now accepts `[min_local, min_global, min_total]`,
  defaulting to `[0, 0, 1]`; all three minima must pass.
- Global picker identities now include their training dataset, such as SAR_CEED,
  independently of the architecture used to load each model.
- Renamed inference picker groups to Local (formerly POS_NEG) and Global
  (formerly POS), including config keys, global config filenames, and QC labels.
  Historical phase/QC and timing labels remain readable; checkpoint files are
  not renamed.
- Continuous AI-PAL picking now combines local POS_NEG and mixed-trained CEED
  POS SAR/PHN models. `picker_group_min_picker_support` defaults to one vote
  across the combined ensemble; local, AWS, and realtime workflows are aligned.
- Existing CEED positive Zarr stores can now import negatives from annual local
  archives without rebuilding positives, using the special-purpose
  `benchmark_picker/train_pos/2.3_copy_negatives_ceed.py` launcher.
- Added separate local and CEED training stages; CEED training now mixes CEED positives with local negatives using explicit per-model batches. The positive-only benchmark remains unchanged.

- Training batches now specify `[positive, negative]` counts directly:
  SAR/FT `[128, 32]`, PHN/RUN `[128, 16]`. Removed ratio-based negative
  reduction; only requested negative windows are loaded. POS-only training
  uses the first count; validation sampling remains unchanged.

- Added standalone `polarity_RUN`: a residual U-Net for 2.5-second vertical
  waveforms with [Noise, Up, Down] soft labels and strict probability > 0.5
  decoding. Source layout follows phase RUN; training-data construction and
  executable workflows are deferred.

- POS picker trigger thresholds now default to 0.6 for SAR/FT and 0.3 for
  PHN/RUN. SoCal AWS FT/RUN configs
  use the current compact architectures; matching checkpoints are required.

- Unified configurable band/location selection (band first by default) and
  fullfed-derived, time-dependent `NET.STA.BAND.LOC` gain inventories across
  local and AWS PAL, AI-PAL realtime, and PALM MFT. Simplified files remain
  supported; routine post-merge station-file reconciliation is unnecessary.

- Documented station-file formats and workflow restrictions; corrected AWS
  inference examples to use SCEDC epoch metadata and matching example dates.

- Added per-picker P/S probability histogram inspection from final QC tables,
  with POS/POS_NEG selection, time filtering and downloadable bin counts.

- Offline final-event PNGs now use `event_waveform_plot/`, distinct from
  filtered waveform data in `event_waveforms/`; existing folders are retained.

- Published AI-PAL final phases now separate detailed per-picker QC into
  matching `.picker_qc.csv` files, including median P/S probabilities across
  random-window votes. The phase row drops the packed picker-uncertainty field;
  built-in readers support both old and new formats.

- Unified offline initial/final range phase and catalog outputs across local
  and AWS combined/split workflows. Offline initial results use
  `2.1_phase_init_AI-PAL`; new intermediate files live under `_internal`.
  Legacy resume paths remain supported; realtime numbering is unchanged.

- Fixed PAL event merging that discarded valid negative magnitudes. Missing
  magnitudes now use `nan`, not the physically valid value -1. Existing
  affected catalogs require magnitude recalculation; they are not auto-repaired.

- Added realtime preferred/reference catalog inspection: FMD, side-by-side
  event maps, matched phase-count plots, residual-ranked one-to-one event
  matching, and waveform copies for detections without eligible counterparts.
  Competing assignments are reported separately rather than labeled as misses.
  Event waveform filenames now include OT, latitude, and longitude to avoid
  overwriting distinct co-OT detections; QC also supports legacy filenames.

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
