"""SCEDC S3 adapter for buffered offline AI-PAL inference."""

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime

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


def get_sta_dict(station_file, when):
    """Return active epoch metadata for association and station selection."""
    return scedc.get_sta_dict_aws(station_file, when)


def _metadata_key(metadata):
    return (
        metadata["net_sta"], metadata["band"],
        tuple(metadata["gains"]), metadata["start"], metadata["end"],
    )


@lru_cache(maxsize=5)
def _daily_records(station_file, date_text):
    observed = UTCDateTime(date_text)
    active = scedc.get_sta_dict_aws(station_file, observed)
    selected = scedc.get_data_dict_aws(
        observed,
        active,
        _client(),
        bucket=os.environ.get("SCEDC_BUCKET", "scedc-pds"),
        root_prefix=os.environ.get(
            "SCEDC_ROOT_PREFIX", "continuous_waveforms"
        ),
    )
    decorated = {}
    for net_sta, records in selected.items():
        metadata = active[net_sta]
        decorated[net_sta] = [
            dict(record, station_metadata=metadata) for record in records
        ]
    return decorated


def get_data_dict(
    date, station_file, normalize_to_three_channels=True,
):
    """Return selected SCEDC objects for one station-file epoch and UTC day."""
    del normalize_to_three_channels
    path = str(Path(station_file).expanduser().resolve())
    return {
        net_sta: [dict(record) for record in records]
        for net_sta, records in _daily_records(
            path, str(UTCDateTime(date).date)
        ).items()
    }


def get_buffered_data_dict(
    date, station_file, buffer_seconds=60.0,
    normalize_to_three_channels=True,
):
    """Return current-day selections plus adjacent-day S3 buffer objects."""
    current = get_data_dict(
        date, station_file,
        normalize_to_three_channels=normalize_to_three_channels,
    )
    if float(buffer_seconds) <= 0:
        return current
    buffered = {net_sta: list(records) for net_sta, records in current.items()}
    for offset in (-1, 1):
        nearby = get_data_dict(
            UTCDateTime(date) + offset * 86400,
            station_file,
            normalize_to_three_channels=normalize_to_three_channels,
        )
        for net_sta in buffered:
            buffered[net_sta].extend(nearby.get(net_sta, []))
    return buffered


def load_station_stream(
    date, station_file, net_sta, normalize_to_three_channels=True,
):
    """Load one buffered station stream through the inference adapter."""
    buffer_seconds = float(os.environ.get("AI_PAL_DATA_BUFFER_SEC", "60"))
    records = get_buffered_data_dict(
        date,
        station_file,
        buffer_seconds=buffer_seconds,
        normalize_to_three_channels=normalize_to_three_channels,
    ).get(net_sta, [])
    return read_data(
        records,
        get_sta_dict(station_file, date),
        start_time=UTCDateTime(date) - buffer_seconds,
        end_time=UTCDateTime(date) + 86400 + buffer_seconds,
        normalize_to_three_channels=normalize_to_three_channels,
    )


def _merge_gain_corrected_streams(streams, start_time, end_time):
    by_component = defaultdict(list)
    for stream in streams:
        for trace in stream:
            component = trace.stats.channel[-1]
            trace = trace.copy()
            trace.stats.location = ""
            trace.stats.channel = "AI" + component
            by_component[component].append(trace)
    output = Stream()
    for component in "ENZ":
        traces = by_component.get(component, [])
        if not traces:
            return Stream()
        target_rate = float(np.median([
            trace.stats.sampling_rate for trace in traces
        ]))
        for trace in traces:
            if float(trace.stats.sampling_rate) != target_rate:
                scedc._interpolate_trace(trace, target_rate)
        merged = Stream(traces=traces)
        merged.merge(method=1, fill_value=0)
        if len(merged) != 1:
            return Stream()
        trace = merged[0]
        if start_time is not None or end_time is not None:
            trace.trim(
                UTCDateTime(start_time) if start_time is not None
                else trace.stats.starttime,
                UTCDateTime(end_time) if end_time is not None
                else trace.stats.endtime,
                nearest_sample=True,
            )
        if not len(trace):
            return Stream()
        output += trace
    return output


def read_data(
    records, stations, start_time=None, end_time=None,
    normalize_to_three_channels=True,
    to_prep=True,
    location_priority=None,
    channel_priority=None,
):
    """Read buffered records with the gain active for each record's day."""
    del stations, normalize_to_three_channels, location_priority, channel_priority
    if not to_prep:
        raise ValueError("AWS raw waveform objects require to_prep=True")
    if not records:
        return Stream()
    try:
        grouped = defaultdict(list)
        metadata_by_key = {}
        for record in records:
            metadata = record["station_metadata"]
            key = _metadata_key(metadata)
            grouped[key].append(record)
            metadata_by_key[key] = metadata
        streams = []
        for key, epoch_records in grouped.items():
            streams.append(scedc.read_data_aws(
                epoch_records,
                metadata_by_key[key],
                _client(),
                bucket=os.environ.get("SCEDC_BUCKET", "scedc-pds"),
                acceleration_instrument_codes=tuple(
                    os.environ.get("ACCELERATION_CODES", "N")
                ),
                start_time=start_time,
                end_time=end_time,
            ))
        return _merge_gain_corrected_streams(
            streams, start_time, end_time
        )
    except Exception as exc:
        station = records[0].get("net_sta", "unknown")
        print(
            "ERROR reading buffered SCEDC data for {}: {}".format(
                station, exc
            ),
            flush=True,
        )
        return Stream()
