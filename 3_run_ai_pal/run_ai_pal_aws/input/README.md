# Station Metadata

The AWS launchers reuse the full SCEDC station file supplied in
`../../../1_run_pal/run_pal_aws/input/station_scedc_aws_selected_20200101_20260701_pal.csv`
(relative to this input directory). Set `FULL_STATION_FILE` to a custom
nine-column NET.STA.BAND epoch file for another deployment.

`example_pal_format1.sta` is retained as a legacy local-format example only.
It has five columns and is **not compatible with the SCEDC AWS reader**.
See [Station File Formats](../../../STATION_FORMATS.md).
