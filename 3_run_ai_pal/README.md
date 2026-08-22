# Run AI-PAL

Unified continuous inference and PAL-association entry points:

- `run_ai_pal_local/`: offline continuous picking and association examples.
- `run_ai_pal_realtime/`: restart-safe multi-picker realtime processing.
- `run_ai_pal_aws/`: reserved for the AWS continuous inference implementation.

Fixed positive-event and negative-event experiments belong to
`../benchmark_picker/`.

The packaged inference workflows default to `CASE_CODE = "eg"`, where `eg` means "example".
Copied case launchers also expose `AI_PAL_ROOT`, defaulting to `~/software/AI-PAL`; set it to the installed package without moving source modules into the case workspace.

## Local Workflow

Each inference launcher defines one `CASE_CODE`. Shared/model config paths,
checkpoint roots, picks, phases, and local realtime outputs are derived from
it; the staged canonical import remains `config_ai_pal`.
`run_ai_pal_local/config_ai_pal_eg.py` supplies shared preprocessing,
sliding-window consensus, picker-ensemble, and PAL parameters. Each
`config_<model>_eg.py` supplies only model and inference settings.

Select local continuous models with `picker_pos_neg_group` and
`picker_pos_group`, and postprocessing models with `repicker_pos_neg_group`
and `repicker_pos_group`, in `run_ai_pal_local/config_ai_pal_<case>.py`.
Local workflows do not run a reference-picker branch. Continuous models must
be subsets of their corresponding repicker groups; loaded objects are reused
during postprocessing. Set either continuous group list to `[]` to disable it;
the other non-empty group then supplies `picks_ENSEMBLE` by itself. POS is
disabled for continuous picking by default but remains enabled for repicking.
The launchers keep `PICKERS_POS_NEG` and `PICKERS_POS` as
path/device registries. Pos+neg checkpoints use the latest `.ckpt` in each
model directory, while positive-only checkpoints use explicit files. Use
`gpu_idx = -1` to run a model on CPU; nonnegative values select that CUDA
device. `NUM_WORKERS`
controls concurrent station-date reading and preprocessing; daily output files
are written by the main thread in deterministic station order. The example
delegates reusable loading, device grouping, shared preprocessing, and ensemble
logic to `PAL_src/offline_picker_runner.py`.
The local ensemble runner reads, merges, gain-corrects, and preprocesses each
station waveform once. A `PreparedPickerStream` then caches one base tensor on
CPU and lazily transfers exactly one copy to each distinct configured device.
Every picker model is loaded once. Per-device locks serialize station inference
on the same GPU while allowing CPU I/O/preprocessing to run ahead; distinct
devices run concurrently. Models assigned to the same device run sequentially
and reuse that device's tensor. Because a buffered daily three-channel stream
can be large, increase `NUM_WORKERS` with attention to host RAM. Each picker
keeps standalone preprocessing as its default when called without a prepared
stream.

For each target UTC date, the local runners read `data_buffer_sec` (default 60 s)
from the preceding and following date folders before preprocessing. Filtering
uses a taper capped by `taper_max_length_sec` (default 10 s), and inference
excludes that same taper duration at both ends of the buffered
stream. Only picks whose P arrival is inside the target UTC date are written to
that date's file, so midnight signals have context without duplicate ownership.
Each model first clusters raw P/S pairs across its sliding windows. Both P and S
arrivals must match within `tp_dev` and `ts_dev`, and each distinct window casts
at most one vote. `picker_min_cluster_size` controls the required number of
windows (default 2). P/S arrival times and probabilities are medians; their
population standard deviations are written as `tp_std`, `ts_std`,
`p_prob_std`, and `s_prob_std`.

After all models finish, daily picks are clustered inside POS_NEG and POS using
`picker_pos_neg_group_min_picker_support` and
`picker_pos_group_min_picker_support`. Accepted group products are then merged
with equal group weight. The canonical result is
written to `output/<CASE_CODE>/picks_ENSEMBLE/`. Set
`save_individual_picker_outputs = False` to use temporary model branches and
retain only the ensemble; `True` preserves POS_NEG `picks_<MODEL>/` and
positive-only `picks_POS-<MODEL>/` branches as well.

Pick rows preserve the original first six fields and append uncertainty and
provenance:

```text
net_sta,tp,ts,s_amp,p_prob,s_prob,tp_std,ts_std,p_prob_std,s_prob_std,num_support,pickers,picker_cluster_sizes
```

`s_amp` is the three-component vector displacement amplitude in metres.
Continuous picking measures it from the gain-corrected, filtered velocity
waveform over `tp - amp_win[0]` through `ts + amp_win[1]`. Event repicking
defers this measurement until PAL reassociation and both-group QC have selected
the final event candidates, so rejected repicker pairs incur no amplitude work.
Strong-motion `HN*` acceleration is converted to velocity before filtering.
PAL then recalculates event magnitude from these final amplitudes and the final
location; incomplete or invalid amplitude windows are excluded.

For individual files, `num_support` counts sliding windows. For ensemble files,
it counts distinct pickers and `pickers` contains their `|`-joined names. The
trailing `picker_cluster_sizes` field always retains each model's original
sliding-window support, for example `SAR:4|FT:3|PHN:5|RUN:2`.
Cross-picker standard deviations are calculated from one equally weighted vote
per contributing picker; within-model standard deviations are retained only in
individual outputs.

Both local picking launchers resume at daily-file granularity when
`OVERWRITE_PICKS = False`. A complete ensemble file is skipped; when individual
outputs are enabled, all selected model files must also exist. If the
individual files are complete but the ensemble is missing, only the ensemble
merge is rerun. A missing complete file or stale `.partial` file causes that
whole UTC day to be rerun, and atomic replacement prevents a partial file from
being mistaken for a completed result. In the one-click workflow, an existing
pick day still has its filtered waveform context rebuilt when positive
repicking is enabled, but continuous-picker inference is skipped. Set
`OVERWRITE_PICKS = True` after changing checkpoints or picker settings.

Run `run_ai_pal_local/2.2_run_pal_assoc_eg.py` after picking. It associates only
the canonical `picks_ENSEMBLE` branch. The final station rows in the phase file
retain probabilities, all four ensemble standard deviations, picker support
count, picker names, and per-picker sliding-window cluster sizes.

### Combined Local Workflow

`run_ai_pal_local/1_run_ai_pal_pick_assoc_eg.py` performs native multi-picker
inference, picker consensus, buffered PAL association, subnet merging, and
range-level catalog/phase assembly in one process. It loads every model once
and still performs picking on buffered daily streams. For a requested target
range it also picks one halo date before and after the range.

Association is performed in non-overlapping intervals controlled by
`association_interval_sec` (3600 s by default). Every interval receives picks
from `association_buffer_sec` before and after its boundaries, but its final
phase/catalog contains only events whose origin time is in the half-open
interval `[start, end)`. All intervals except the final one of a UTC day can be
completed as soon as that day's picks exist. The final interval waits for the
next day's picks, so boundary events retain forward context.

The combined workflow retains only the preprocessed, filtered station waveforms
after daily inference. Raw streams are released after picking. After the ready
intervals are associated, the full filtered day is replaced with an owned copy
of only its final interval. While the next day is read, the retained filtered
waveform footprint is therefore approximately one day plus one hour. Once the
previous final interval is completed, its waveform is released. This retention
is opt-in and does not change the memory behavior of
`2.1_run_ai_pal_pick_eg.py`.

`NUM_PICK_WORKERS` controls concurrent station-date reading/preprocessing.
`NUM_ASSOC_WORKERS` controls concurrently processed hourly association
intervals. Subnets are processed sequentially inside each hour, while PAL runs
for different hours concurrently. Optional dual-group repicking/reassociation
is serialized through the loaded event-model ensembles. The combined
workflow writes halo-day pick files because they provide boundary context, but
its final catalog and phase files include only dates in `TIME_RANGE`. Raw subnet
results and status files are written below `phase_ENSEMBLE_PAL/hourly_assoc`.
Hourly origin-time-owned phase, catalog, and event-group files are written to
`phase_ENSEMBLE_PAL_final`; daily and range-level concatenations are also kept
there.

Set `enable_post_process = True` to run event-based postprocessing after each
hourly association. Combined workflows defer displacement amplitude and PAL
glitch QC until initial association selects phase pairs; events falling below
`min_sta` after glitch rejection are removed. Final amplitudes and magnitudes
are measured only for picks accepted by post-reassociation.
`PICKERS_POS_NEG` contains SAR, PHN, FT, and RUN trained
on positive and negative samples; `PICKERS_POS` contains their positive-only
counterparts. Continuous SAR/PHN instances are selected from POS_NEG and reused,
so combined and realtime processing holds eight native model copies rather
than ten. For every event, all available stations through the epicentral
distance of its farthest initial phase are evaluated. Only P/S pairs detected
by both POS_NEG and POS groups are associated with the full-network PAL
parameters, and every qualified PAL candidate is retained. For each candidate,
theoretical P/S arrivals are recalculated from its updated origin and location.
For stations not already represented by an anchor, POS_NEG-only and POS-only
pairs are then supplemented when both arrival residuals are within `tp_dev`
and `ts_dev`. If the anchor picks yield no PAL candidate,
the preliminary event is discarded. Subnet membership and a separate
post-repick station-count test are not used at this stage. Events that converge
after reassociation are
collapsed immediately in the local hourly product using the configured origin,
location, depth, and shared-phase merge criteria. Daily and range-level files
therefore concatenate already deduplicated hourly results.
Because hourly association uses buffered picks, the first subnet merge keeps
all candidate events without filtering on preliminary origin time. The
half-open hourly ownership filter `[hour_start, hour_end)` is applied only
after repicking and reassociation, using the updated origin time.

The postprocessor draws `repick_num_repeat` randomized 25 s windows once per
station-event job and shares them across both repicker groups. Windows from
multiple jobs fill `repick_batch_size`; CPU slicing uses the workflow worker
count. Models assigned to one device run sequentially against each resident
batch, while different device groups run concurrently. When possible,
theoretical P is at least
`repick_phase_buffer_sec` after the window start and theoretical S is that far
before its end. If the P-S span leaves insufficient margin, the available
`25 - (ts - tp)` shift range is used. Initial P/S arrival values are discarded:
they neither constrain repicker output nor enter PAL reassociation. Every
vote-supported pair is retained as a candidate. Both-group pairs anchor PAL;
single-group pairs are tested only after PAL updates the event origin and
location. Predictions are clustered across random
windows within each model and then independently within POS_NEG and POS by
both P and S. One station can contribute multiple alternative pairs to PAL.
`repick_group_min_picker_support` (default 2) is applied separately to each
group. When both groups agree, POS supplies the reported time and probability.
Within each model, the minimum repeated-window support is
`ceil(repick_num_repeat * repick_min_window_vote_ratio)`. The default ratio is
`0.2`, so 20 randomized windows require four matching votes.

For intervals that need adjacent-day waveform context, only the theoretical
event span plus one event-repicker window on each side is copied and merged;
the full neighboring daily stream is not duplicated for repicking.

The former near-source distance exception and minimum refined-pick ratio are
not used. Reliability is established structurally by associating only
both-group anchors. Reference picker branches are unchanged.
The final station row keeps the usual
ensemble standard deviations and adds column 14,
`picker_uncertainties`, formatted with group-qualified model names such as
`POS_NEG:SAR:tp=...|POS:SAR:tp=...`, containing per-model arrival-time
and probability standard deviations. The filtered waveform segments are merged
before event windows are sliced, including next-day data for the final hour.
Column 15, `pick_provenance`, records only `both_groups`, `pos_neg_only`, or
`pos_only`. Columns 16-19 record repicker diagnostics:
`repick_status`, agreeing-group count (`repick_support`), group names
(`repick_sources`), and required group count. Per-model randomized-window votes
remain in `picker_cluster_sizes`. Events without enough repicker-derived pairs
for PAL reassociation are discarded; initial continuous-picker pairs are never
used as fallback output.

Columns 20-22 are `p_snr_e`, `p_snr_n`, and `p_snr_z`. They are measured only
for final reassociated event picks, using PAL's energy STA/LTA definition on
the same filtered, gain-corrected E/N/Z velocity waveforms used for repicking.
For each component, the reported value is the maximum ratio from 0.5 s before
through 1.0 s after P, with PAL's 0.8 s forward STA and 6.0 s preceding LTA.
Missing waveform coverage is reported as `-1`.

Set `save_filtered_event_waveforms` to serialize the same filtered event spans
used by the repicker as three SAC files per station after final reassociation
and merging. The underlying `hour_complete_callback` remains available for
additional event products before retained waveform arrays are released.

`2.1_run_ai_pal_pick_eg.py` uses one `FULL_STATION_FILE` containing every station to
pick. It reads the configured adjacent-day waveform buffer but writes only P/S
pairs whose P arrival belongs to the target UTC date. Its canonical output is
`output/<CASE_CODE>/picks_ENSEMBLE`, matching the one-click workflow.

In `2.2_run_pal_assoc_eg.py`, leave `SUBNET_STATION_FILES` empty to
associate that complete list once with the `full` parameters. Optional subnet
files map in order to the configured keys after `default` and `full` (`r1`,
`r2`, ...), run independently, and are merged afterward.

`2.3_run_ai_pal_repick_reassoc_eg.py` reads the daily initial detections from
`phase_ENSEMBLE_PAL/daily_assoc/merged`. It performs the same dual repicker-group
consensus, full-network PAL reassociation, candidate retention, and duplicate
merge as the one-click local workflow. Raw data are read with ObsPy time bounds
only for event-selected stations. When a requested span crosses midnight, the
needed files from both date folders are read and merged before gain correction
and shared preprocessing. A `taper_max_length_sec` halo is included on both
sides of each requested span and removed after filtering.
Set `enable_post_process = True` before running this dedicated postprocessing
stage. The outer loop is over all initial events, while `NUM_WORKERS` controls
concurrent waveform I/O and preprocessing for stations required by the current
event. Repicker models are loaded once and reused for the full event
sequence. With `OVERWRITE = False`, completed daily phase files are resumed;
use `OVERWRITE = True` after changing repick settings or enabling a new event
waveform product.
The launcher reports range-wide initial-event progress every 1,000 completed
events, including elapsed time, throughput, ETA, and `num_done/num_all`.

The staged postprocessor writes only one final phase file per UTC day below
`phase_ENSEMBLE_PAL_final`; it does not retain hourly, catalog, or range-level
phase products. Set `enable_event_waveform_plot` to publish
final reassociated event plots in `event_waveform`. Set
`save_filtered_event_waveforms` independently to write one directory per final
event under `event_waveforms`, containing three filtered SAC files per station.
SAC headers `o`, `t0`, and `t1` record the event origin and final P/S arrivals;
the station-file layout is compatible with the event waveform input expected by
the cross-correlation relocation preprocessors in `4_location/2_hypodd`.

Fixed positive-event and negative-event launchers are maintained in
`../benchmark_picker/` because they are evaluation workflows rather than
continuous-data processing.
## AWS Workflow

The AWS multi-picker inference launcher remains to be implemented. The shared
native picker, ensemble, PAL association, and phase-merge sources already use
the same 13-column QC schema, so AWS jobs built on these modules preserve
`picker_cluster_sizes` identically to local and realtime runs.
## Realtime Workflow

Edit `run_ai_pal_realtime/config_ai_pal_eg.py` for shared preprocessing,
association, merging, realtime-loop parameters, and the `picker_*_group` and
`repicker_*_group` selections. These config lists select models.
The launcher's `PICKERS_POS_NEG`, `PICKERS_POS`, and `PICKER_REF`
dictionaries are available-model
registries and normally change only for paths or devices. The realtime folder contains
model configs for SAR, FT, PHN, RUN,
and SeisBench PHN-SB. Reference configs follow
`config_ref_<picker>_<case>.py`; for example,
`config_ref_phn-sb_eg.py`. Its `win_len` and `overlap_sec` are expressed in
seconds, with `win_len * samp_rate == 3000`; one `trig_thres` controls both
P and S. Paths, per-model CPU/GPU assignments, station files, and checkpoint
roots remain in `run_ai_pal_realtime/1_run_ai_pal_pick_assoc_realtime_eg.py`.
Native and postprocessing specifications share `PICKERS_POS_NEG` and
`PICKERS_POS`; reference specifications are in `PICKER_REF`. All tunable
settings precede the connection-code boundary in that launcher.
Realtime source miniSEED windows are read as delivered, with no adjacent-file
waveform buffer. Their canonical `T0` and `T1` are the median start and
exclusive end times of all traces after sampling-rate cleanup and ObsPy merge.
The valid event-origin interval is
`[T0 + taper_max_length_sec, T1 - taper_max_length_sec - association_buffer_sec)`.
The association buffer is an output hard limit here, not extra waveform input.
The shared picker runtime disables PyTorch's unsupported NNPACK CPU backend
before model loading; this implementation detail is not user-configurable.
For each realtime station, component selection and gain conversion are followed
by one shared `PreparedPickerStream` preprocessing pass. Native AI-PAL pickers
reuse one cached tensor per distinct device; per-device locks allow station
preprocessing to remain parallel while inference is serialized on each device
and concurrent across different devices. The SeisBench reference adapter uses
the same filtered, edge-trimmed stream, but SeisBench `classify` retains control
of its own batching and device transfer.

`FULL_STATION_FILE` normally controls the stations prepared for all pickers.
Set it to `None` to build the picking list dynamically as the selector-deduplicated
union of `SUBNET_STATION_FILES`. In that mode at least one subnet file is required,
and the generated union is stored below the output `_internal/stations` directory.
`SUBNET_STATION_FILES` also controls PAL association. Leave it empty with a full
file to associate that list once with the `full` parameter override. When subnet
files are supplied, their names select per-subnet overrides and inherit all
unspecified values from `subnet_assoc_params["default"]`. Cross-subnet merge
parameters are retained in the shared config but are bypassed in full-only
mode.

The default first picker group (`SAR` and `PHN`) is merged with
equal-weight joint P/S consensus and then associated once. Pickers in later
reference groups remain independent picking and PAL branches.

Set `enable_post_process = True` to apply the same dual-group postprocessing
used by the combined local workflow. Repicking runs only on the
preferred `AI-PAL` branch built from `picker_groups[0]`; reference branches are
unchanged. Each source segment is first associated by subnet and merged across
subnets. Its detections are then repicked and reassociated with full-network
PAL, merged for duplicates within that same segment, and filtered to the
corrected origin-time interval. Repicking uses only one current realtime source segment; it
does not merge waveforms across source windows or retain an additional
raw-waveform copy. If the latest theoretical S arrival among the
eligible stations is later than
`source_end_time - taper_max_length_sec`, the entire event is left unchanged
rather than waiting for the next segment. No event merge is performed between
realtime source segments.

Realtime `PICKERS_POS_NEG` and `PICKERS_POS` entries name one
explicit `ckpt` file per model. Packaged Cent-Cal positive-plus-negative
checkpoints are in `Pre-trained_models/Cent-Cal_ckpt/`, while CEED positive-only
checkpoints are in `Pre-trained_models/CEED_ckpt/`. Set launcher paths to those
package files or to deployment-managed copies.
Each `config_<model>_pos_<case>.py` is a self-contained model config rather
than a subclass of its continuous counterpart. The packaged values currently
match, but architecture, training, and inference settings can diverge for
future positive models without changing the continuous picker config.

On restart, `_internal/event_repick/event_repick_status.csv` identifies final
source segments already repicked and reassociated. Missing segments are
supplemented from their merged subnet phase files and source MiniSEED files by
running shared preprocessing only; continuous inference and initial subnet PAL
association are not repeated. Corrected segment products and any unreported
origin-time tails are generated only after this supplementation. The eight native repicker models load before
waveform ingest; continuous SAR/PHN objects are reused by the POS_NEG group
instead of loaded twice.

Realtime filtered waveforms remain available until that source segment has
completed optional repicking and reassociation. During initial picking they
use a disposable disk-backed `_internal/waveform_cache/<segment>` cache rather
than accumulating every station's full filtered stream in host RAM. After
subnet merging, only event-relevant stations are loaded; the exact shared
preprocessing result is therefore still used for repicking and waveform plots.
Cache files are removed before polling resumes. Host RSS, total cgroup memory,
and CUDA allocated/reserved memory are recorded during station progress and at
the realtime stage boundaries.

When `enable_event_waveform_plot` is enabled, short filtered waveform snapshots
are retained until the segment advances the external reporting cursor. The
first segment reports its full corrected interval. Each later overlapping
segment reports only `[previous_T1_corr, current_T1_corr)`, while its internal
phase file retains the full corrected segment interval. Only these newly
reported events are plotted to
`OUT/event_waveform_final_AI-PAL/<origin-time>.png`. Set
`enable_event_waveform_plot_ref` to zero-based indexes into the flattened
reference groups in `picker_groups[1:]`; those final reference detections are
written separately under
`OUT/event_waveform_final_ref_<picker>_PAL/<origin-time>.png`.
Snapshots earlier than the committed interval end are released immediately
after plotting. Plotting uses the window
from 5 s before origin through 10 s after the latest S pick and writes PNGs at
200 DPI. Filtered three-component station waveforms are
independently normalized, drawn in three separate gray component lanes per
station, and ordered from smaller to larger
epicentral distance. P and S markers use solid/dashed lines; retained initial,
POS_NEG+POS, POS_NEG-only, and POS-only pairs use distinct colors. The x axis is seconds relative to
the event origin time.

The single launcher performs startup backfill, multi-picker inference, parallel
subnet PAL association, subnet merging, repicking/reassociation, within-segment
deduplication, corrected-origin filtering, monotonic-tail publication,
heartbeat logging, and continuous polling.
