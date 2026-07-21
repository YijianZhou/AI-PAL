#!/usr/bin/env python3
"""SCEDC S3 data access and epoch-aware PAL station metadata."""

from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from obspy import Stream, UTCDateTime, read


WAVEFORM_NAME = re.compile(
    r"^(?P<net>.{2})(?P<sta>.{5})(?P<chn>.{3})(?P<loc>.{2})_?"
    r"(?P<year_day>\d{7})\.ms$"
)
COMPONENT_ORDER = ("E", "N", "Z")


def _as_date(value):
    if isinstance(value, date):
        return value
    if isinstance(value, UTCDateTime):
        return value.date
    text = str(value).strip()
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def _component(channel):
    return {"1": "E", "2": "N"}.get(channel[-1], channel[-1])


def _normalize_location(value):
    value = value.strip("_ ")
    return value if value else "--"


@lru_cache(maxsize=8)
def _load_station_epochs(station_file):
    epochs = defaultdict(list)
    path = Path(station_file).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as fp:
        for line_number, row in enumerate(csv.reader(fp), start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 PAL fields, got {len(row)}"
                )
            codes = row[0].strip().split(".")
            if len(codes) != 3:
                raise ValueError(
                    f"{path}:{line_number}: first field must be NET.STA.BAND"
                )
            net, sta, band = codes
            start, end = _as_date(row[7]), _as_date(row[8])
            if start >= end:
                raise ValueError(f"{path}:{line_number}: t0 must be before t1")
            epoch = {
                "net": net,
                "sta": sta,
                "net_sta": f"{net}.{sta}",
                "band": band,
                "latitude": float(row[1]),
                "longitude": float(row[2]),
                "elevation": float(row[3]),
                "gains": tuple(float(value) for value in row[4:7]),
                "start": start,
                "end": end,
            }
            epochs[epoch["net_sta"]].append(epoch)

    for net_sta in epochs:
        epochs[net_sta].sort(key=lambda item: (item["start"], item["end"]))
    return dict(epochs)


def get_sta_dict_aws(station_file, when):
    """Return active NET.STA metadata selected from NET.STA.BAND epochs.

    Intervals use the same half-open convention as the station file: t0 <= day < t1.
    """
    observed_date = _as_date(when)
    active = {}
    for net_sta, epochs in _load_station_epochs(str(Path(station_file).resolve())).items():
        matches = [
            epoch for epoch in epochs
            if epoch["start"] <= observed_date < epoch["end"]
        ]
        if len(matches) > 1:
            descriptions = ", ".join(
                f"{item['band']}:{item['start']}/{item['end']}" for item in matches
            )
            raise ValueError(
                f"overlapping active station epochs for {net_sta} on "
                f"{observed_date}: {descriptions}"
            )
        if matches:
            active[net_sta] = matches[0]
    return active


def to_associator_sta_dict(active_sta_dict):
    """Convert AWS metadata to the list layout expected by PAL's associator."""
    return {
        net_sta: [
            row["latitude"], row["longitude"], row["elevation"], list(row["gains"])
        ]
        for net_sta, row in active_sta_dict.items()
    }


def build_s3_client(region="us-west-2", access_mode="unsigned"):
    config = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
    if access_mode == "unsigned":
        config["signature_version"] = UNSIGNED
    return boto3.client("s3", region_name=region, config=BotoConfig(**config))


def _parse_key(key):
    match = WAVEFORM_NAME.match(key.rsplit("/", 1)[-1])
    if match is None:
        return None
    net = match.group("net").strip("_ ")
    sta = match.group("sta").strip("_ ")
    channel = match.group("chn").strip("_ ")
    if not net or not sta or len(channel) != 3:
        return None
    return {
        "key": key,
        "net": net,
        "sta": sta,
        "net_sta": f"{net}.{sta}",
        "location": _normalize_location(match.group("loc")),
        "channel": channel,
        "band": channel[:2],
        "component": _component(channel),
    }


def _location_rank(location, location_priority):
    if location in location_priority:
        return (0, location_priority.index(location))
    if location != "--":
        return (1, location)
    return (2, location)


def _choose_component_record(records):
    # Lettered orientations are preferred to equivalent numeric orientations.
    return min(records, key=lambda row: (row["channel"][-1] in "12", row["key"]))


def get_data_dict_aws(
    when,
    active_sta_dict,
    s3_client,
    bucket="scedc-pds",
    root_prefix="continuous_waveforms",
    location_priority=("10", "20", "01", "02", "00", "--"),
):
    """List one SCEDC day and select the requested band for active stations.

    Values contain exactly three E/N/Z objects, or one selected object marked
    for three-component expansion. One- and two-component groups use the
    fallback trace selected below.
    """
    observed_date = _as_date(when)
    doy = observed_date.timetuple().tm_yday
    prefix = f"{root_prefix}/{observed_date.year}/{observed_date.year}_{doy:03d}/"
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            record = _parse_key(item["Key"])
            if record is None or record["net_sta"] not in active_sta_dict:
                continue
            if record["band"] != active_sta_dict[record["net_sta"]]["band"]:
                continue
            grouped[record["net_sta"]][record["location"]][record["component"]].append(
                record
            )

    selected = {}
    for net_sta, by_location in grouped.items():
        # Match the inventory selector: choose location first, then use that
        # location's selected band. Do not fall through to a different location.
        location = min(
            by_location,
            key=lambda value: _location_rank(value, tuple(location_priority)),
        )
        by_component = by_location[location]
        components = set(by_component)
        if len(components) >= 3:
            assigned = {
                component: _choose_component_record(by_component[component])
                for component in COMPONENT_ORDER if component in by_component
            }
            extras = [
                _choose_component_record(by_component[component])
                for component in sorted(components - set(assigned))
            ]
            for component in COMPONENT_ORDER:
                if component not in assigned:
                    assigned[component] = extras.pop(0)
            selected[net_sta] = [assigned[component] for component in COMPONENT_ORDER]
        elif len(components) in (1, 2):
            # PAL requires E/N/Z. Prefer the vertical trace; when the available
            # traces are both horizontal, repeat the first E/N trace.
            if "Z" in by_component:
                component = "Z"
            else:
                component = min(
                    components,
                    key=lambda value: (
                        COMPONENT_ORDER.index(value)
                        if value in COMPONENT_ORDER else len(COMPONENT_ORDER),
                        value,
                    ),
                )
            selected[net_sta] = [_choose_component_record(by_component[component])]
    return selected


def _interpolate_trace(trace, sampling_rate):
    """Return a trace sampled on the requested rate while preserving timing."""
    sampling_rate = float(sampling_rate)
    if float(trace.stats.sampling_rate) == sampling_rate:
        return trace
    if len(trace) < 2:
        raise ValueError(
            f"cannot interpolate {trace.id} with only {len(trace)} sample(s)"
        )
    trace.data = np.asarray(trace.data, dtype=np.float64)
    trace.interpolate(
        sampling_rate=sampling_rate,
        method="lanczos",
        a=12,
    )
    return trace

def _read_s3_trace(record, s3_client, bucket):
    body = s3_client.get_object(Bucket=bucket, Key=record["key"])["Body"].read()
    stream = read(io.BytesIO(body), format="MSEED")
    matching = Stream(
        traces=[
            trace for trace in stream
            if trace.stats.network.strip() == record["net"]
            and trace.stats.station.strip() == record["sta"]
            and trace.stats.channel.strip() == record["channel"]
            and _normalize_location(trace.stats.location) == record["location"]
        ]
    )
    if not matching:
        matching = stream
    target_trace = max(
        matching,
        key=lambda trace: float(trace.stats.endtime - trace.stats.starttime),
    )
    target_rate = float(target_trace.stats.sampling_rate)
    for trace in matching:
        _interpolate_trace(trace, target_rate)
    matching.merge(method=1, fill_value=0)
    if len(matching) != 1:
        raise ValueError(
            f"expected one merged trace in s3://{bucket}/{record['key']}, "
            f"found {len(matching)}"
        )
    return matching[0]


def read_data_aws(
    records,
    station_metadata,
    s3_client,
    bucket="scedc-pds",
    acceleration_instrument_codes=("N",),
):
    """Read selected S3 objects, correct gain, and return PAL-ordered E/N/Z.

    SEED instrument code N denotes an accelerometer. Those traces are converted
    from acceleration to velocity after gain correction and before PAL filtering.
    A one- or two-channel station contributes one trace copied to E, N, and Z.
    """
    if len(records) not in (1, 3):
        return Stream()
    source_traces = [_read_s3_trace(record, s3_client, bucket) for record in records]
    if len(source_traces) > 1:
        target_rate = float(np.median([
            trace.stats.sampling_rate for trace in source_traces
        ]))
        for trace in source_traces:
            _interpolate_trace(trace, target_rate)
    if len(records) == 1:
        source_component = records[0]["component"]
        source_gain_index = (
            COMPONENT_ORDER.index(source_component)
            if source_component in COMPONENT_ORDER else 2
        )
        assignments = [(component, source_traces[0].copy(), source_gain_index)
                       for component in COMPONENT_ORDER]
    else:
        assignments = [
            (component, trace, index)
            for index, (component, trace) in enumerate(zip(COMPONENT_ORDER, source_traces))
        ]

    output = Stream()
    for component, trace, gain_index in assignments:
        gain = station_metadata["gains"][gain_index]
        if not np.isfinite(gain) or gain == 0:
            raise ValueError(f"invalid gain for {station_metadata['net_sta']}: {gain}")
        trace.data = np.asarray(trace.data, dtype=np.float64) / gain
        if station_metadata["band"][1:2] in acceleration_instrument_codes:
            trace.detrend("demean").detrend("linear")
            trace.integrate(method="cumtrapz")
            trace.detrend("linear")
        trace.stats.network = station_metadata["net"]
        trace.stats.station = station_metadata["sta"]
        trace.stats.channel = station_metadata["band"] + component
        output += trace
    return output


def get_pal_picks(date_value, pick_dir):
    """Read PAL pick output while retaining its NET.STA identifier."""
    dtype = [
        ("net_sta", "O"), ("sta_ot", "O"), ("tp", "O"),
        ("ts", "O"), ("s_amp", "O"),
    ]
    path = Path(pick_dir) / f"{_as_date(date_value).isoformat()}.pick"
    if not path.exists():
        return np.array([], dtype=dtype)
    picks = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            values = line.rstrip("\n").split(",")
            if len(values) < 5:
                continue
            picks.append(
                (values[0], UTCDateTime(values[1]), UTCDateTime(values[2]),
                 UTCDateTime(values[3]), float(values[4]))
            )
    return np.array(picks, dtype=dtype)
