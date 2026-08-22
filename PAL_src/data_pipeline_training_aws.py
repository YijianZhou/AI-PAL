"""SCEDC S3 station-day loader for AWS training-sample cutting."""

import os

from obspy import Stream

import data_pipeline_aws as scedc


_s3_client = None


def _client():
    global _s3_client
    if _s3_client is None:
        _s3_client = scedc.build_s3_client(
            os.environ.get("SCEDC_REGION", "us-west-2"),
            os.environ.get("SCEDC_ACCESS_MODE", "signed"),
        )
    return _s3_client


def load_station_stream(when, station_file, net_sta):
    """Return one selected, gain-corrected E/N/Z station-day stream."""
    try:
        active = scedc.get_sta_dict_aws(station_file, when)
        metadata = active.get(net_sta)
        if metadata is None:
            return Stream()
        client = _client()
        selected = scedc.get_data_dict_aws(
            when,
            {net_sta: metadata},
            client,
            bucket=os.environ.get("SCEDC_BUCKET", "scedc-pds"),
            root_prefix=os.environ.get(
                "SCEDC_ROOT_PREFIX", "continuous_waveforms"
            ),
            location_priority=(),
        )
        records = selected.get(net_sta)
        if not records:
            return Stream()
        return scedc.read_data_aws(
            records,
            metadata,
            client,
            bucket=os.environ.get("SCEDC_BUCKET", "scedc-pds"),
            acceleration_instrument_codes=tuple(
                os.environ.get("ACCELERATION_CODES", "N")
            ),
        )
    except Exception as exc:
        print(
            "ERROR loading {} {} from SCEDC S3: {}".format(
                when, net_sta, exc
            ),
            flush=True,
        )
        return Stream()
