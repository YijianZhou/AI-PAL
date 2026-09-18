"""GMMA 1.2.12 reference worker; input/output contain no waveform arrays."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyproj import Proj

from phase_merge import write_phase_file
from reference_association import require_gamma_runtime


def utc_datetime(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_pydatetime(warn=False)


def prepare_inputs(records, station_dict, settings):
    config = dict(settings)
    if config.get("use_amplitude", False):
        raise ValueError("GaMMA reference currently requires use_amplitude=False")
    for name in ("ncpu", "oversample_factor", "min_sta"):
        value = config[name]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError("{} must be a positive integer".format(name))
        config[name] = int(value)
    config.update(use_amplitude=False,
                  min_picks_per_eq=2 * config["min_sta"],
                  min_p_picks_per_eq=config["min_sta"],
                  min_s_picks_per_eq=config["min_sta"])
    lat = [info[0] for info in station_dict.values()]
    lon = [info[1] for info in station_dict.values()]
    if not lat:
        raise ValueError("GaMMA requires station metadata")
    projection = Proj(proj="aeqd", lat_0=float(np.median(lat)),
                      lon_0=float(np.median(lon)), datum="WGS84", units="km")
    stations = []
    for key, (latitude, longitude, elevation) in station_dict.items():
        x, y = projection(longitude, latitude)
        stations.append({"id": key, "x(km)": x, "y(km)": y,
                         "z(km)": -elevation / 1000.0})
    stations = pd.DataFrame(stations)
    margin = float(config.pop("xy_margin_deg"))
    latitude_range = config.pop("lat_range") or [min(lat) - margin, max(lat) + margin]
    longitude_range = config.pop("lon_range") or [min(lon) - margin, max(lon) + margin]
    depth_range = config.pop("depth_km")
    for bounds in (latitude_range, longitude_range, depth_range):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("invalid GaMMA search bounds: {}".format(bounds))
    # Sample all edges because extrema need not occur at projected corners.
    edge_lon = np.linspace(*longitude_range, 101)
    edge_lat = np.linspace(*latitude_range, 101)
    xs, ys = projection(
        np.concatenate([edge_lon, edge_lon, np.full(101, longitude_range[0]),
                        np.full(101, longitude_range[1])]),
        np.concatenate([np.full(101, latitude_range[0]), np.full(101, latitude_range[1]),
                        edge_lat, edge_lat]),
    )
    config.update({"dims": ["x(km)", "y(km)", "z(km)"],
                   "x(km)": [float(min(xs)), float(max(xs))],
                   "y(km)": [float(min(ys)), float(max(ys))], "z(km)": depth_range})
    config["bfgs_bounds"] = (config["x(km)"], config["y(km)"], depth_range, (None, None))
    rows = []
    for index, record in enumerate(records):
        if record["net_sta"] not in station_dict:
            raise ValueError("missing GaMMA station: {}".format(record["net_sta"]))
        for phase, time_key, prob_key in (("p", "tp", "p_prob"), ("s", "ts", "s_prob")):
            probability = float(record[prob_key])
            if not np.isfinite(probability) or not 0 < probability <= 1:
                raise ValueError("invalid {} probability: {}".format(phase, probability))
            rows.append({"id": record["net_sta"], "timestamp": pd.Timestamp(record[time_key]),
                         "type": phase, "prob": probability, "source_row": index})
    picks = pd.DataFrame(rows, columns=["id", "timestamp", "type", "prob", "source_row"])
    return picks, stations, config, projection


def paired_events(events, assignments, picks, records, projection, min_sta):
    assigned = {}
    for pick_index, event_index, _score in assignments:
        row = picks.loc[pick_index]
        assigned.setdefault(event_index, {}).setdefault(row["id"], {})[row["type"]] = row
    output = []
    for event in events:
        paired = []
        for station, phases in assigned.get(event["event_index"], {}).items():
            if set(phases) != {"p", "s"}:
                continue
            p_record = records[int(phases["p"]["source_row"])]
            s_record = records[int(phases["s"]["source_row"])]
            if pd.Timestamp(s_record["ts"]) <= pd.Timestamp(p_record["tp"]):
                continue
            paired.append({
                "sta": station, "p": utc_datetime(p_record["tp"]),
                "s": utc_datetime(s_record["ts"]), "score": -1.0, "quality": -1,
                "p_prob": p_record["p_prob"], "s_prob": s_record["s_prob"],
                "tp_std": p_record.get("tp_std", 0), "ts_std": s_record.get("ts_std", 0),
                "p_prob_std": p_record.get("p_prob_std", 0),
                "s_prob_std": s_record.get("s_prob_std", 0),
                "sources": p_record.get("sources", ""), "pick_provenance": "initial",
                "num_support": int(min(p_record.get("num_support", 1), s_record.get("num_support", 1))),
            })
        if len({".".join(p["sta"].split(".")[:2]) for p in paired}) < min_sta:
            continue
        lon, lat = projection(event["x(km)"], event["y(km)"], inverse=True)
        output.append({"time": utc_datetime(event["time"]), "lat": lat, "lon": lon,
                       "depth": event["z(km)"], "mag": -1.0, "picks": paired})
    return output


def run(payload, output_path):
    require_gamma_runtime()
    from gamma.utils import association
    picks, stations, config, projection = prepare_inputs(
        payload["picks"], payload["stations"], payload["config"])
    np.random.seed(42)
    events, assignments = (association(picks, stations, config, method=config["method"])
                           if len(picks) >= config["min_picks_per_eq"] else ([], []))
    output = paired_events(events, assignments, picks, payload["picks"], projection,
                           config["min_sta"])
    path = Path(output_path)
    # Keep every native assignment, including unpaired phases excluded by PAL's paired schema.
    event_columns = ["time", "x(km)", "y(km)", "z(km)", "magnitude", "sigma_time",
                     "sigma_amp", "cov_time_amp", "gamma_score", "num_picks",
                     "num_p_picks", "num_s_picks", "event_index"]
    pd.DataFrame(events, columns=event_columns).to_csv(path.with_suffix(".events.csv"), index=False)
    assigned = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    picks.join(assigned.set_index("pick_index")).to_csv(path.with_suffix(".assignments.csv"), index=False)
    write_phase_file(path, output)
    print("GaMMA: {} native events -> {} paired events; {} assigned phases".format(
        len(events), len(output), len(assignments)))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), sys.argv[2])
