# PAL Source

Rule-based PAL picker, association, phase merging, and reusable orchestration modules.
Executable examples live in `../1_run_pal/`. AI picker source folders are siblings
named `picker_*`; combined AI picking and PAL association workflows live in
`../3_run_ai_pal/`.

`picker_stream.py` owns the shared preprocessed waveform, window geometry,
missing-channel mask, and one lazily created tensor per distinct device.
`pick_ensemble.py` owns both distinct-window P/S-pair consensus and equal-weight
cross-picker consensus. Its extended pick schema is consumed by
`data_pipeline.py`, preserved by `associator_pal.py`, and retained through
subnetwork merging, within-segment phase merging, and corrected realtime
origin-time publication.
