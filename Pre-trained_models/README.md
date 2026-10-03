# Packaged Checkpoints

The inference examples reference this directory through `AI_PAL_ROOT`.
Keep model configurations compatible with the selected checkpoint.

- `SoCal_2020-2025_ckpt/*_best.ckpt`: default local inference models.
- `SoCal_2020-2025_ckpt/realtime_sar_best.ckpt`: preserved realtime SAR
  checkpoint, which is not byte-identical to `sar_best.ckpt`.
- `CEED/CEED_ckpt/ceed_*_best.ckpt`: Global CEED inference models.
- `Cent-Cal_ckpt/`: alternative Central California checkpoints.

AWS stages CEED checkpoints from here into temporary job bundles; trained
Local models continue to come from the configured S3 training run.
Custom work directories may keep their own input checkpoints.
Historical third-party/legacy checkpoints under `References/` are not
active packaged inference defaults and are left untouched.
