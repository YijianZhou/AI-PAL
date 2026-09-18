# AI-PAL Changelog

Major user-facing changes only. Implementation notes and the maintenance
procedure are in [CHANGELOG_INTERNAL.md](CHANGELOG_INTERNAL.md).

## Unreleased

No changes recorded after the v7.1 release baseline.

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
