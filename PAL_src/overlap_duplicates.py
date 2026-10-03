"""Ranked, directed P-to-S interval duplicate suppression for final events."""

import csv
import math
from datetime import timedelta
from pathlib import Path


def station_id(value):
    return '.'.join(str(value).split('.')[:2])


def suppress_overlap_duplicates(events, stations, origin_tol=10.0,
                                overlap_fraction=0.5, failed_fraction=0.5,
                                vp=5.9, vs=3.5):
    """Return unchanged survivors and every directed suppression edge.

    Components are fixed before suppression. Even suppressed parents retain
    outgoing edges; this is a DAG traversal, not iterative survivor clustering.
    Station rows outside the annulus still count in the child's denominator.
    """
    if not math.isfinite(origin_tol) or origin_tol < 0:
        raise ValueError('overlap duplicate OT tolerance must be finite and >= 0')
    if not all(math.isfinite(v) and 0 <= v <= 1
               for v in (overlap_fraction, failed_fraction)):
        raise ValueError('overlap duplicate fractions must be in [0, 1]')
    if not all(math.isfinite(v) and v > 0 for v in (vp, vs)):
        raise ValueError('overlap duplicate velocities must be finite and positive')
    geometry = {}
    for key, values in stations.items():
        xyz = tuple(float(v) for v in values[:3])
        if all(math.isfinite(v) for v in xyz):
            geometry.setdefault(station_id(key), xyz)
    components = []
    for index in sorted(range(len(events)), key=lambda i: events[i]['time']):
        if (not components or
                (events[index]['time'] - events[components[-1][-1]]['time'])
                .total_seconds() > origin_tol):
            components.append([])
        components[-1].append(index)
    rejected = set()
    edges = []

    def rank(index):
        event = events[index]
        mag = float(event['mag'])
        # -1 is the published missing-magnitude sentinel, not a ranking value.
        if not math.isfinite(mag) or mag == -1:
            mag = -math.inf
        return (-len({station_id(p['sta']) for p in event['picks']}),
                -mag, event['time'], index)

    for component_id, component in enumerate(components):
        ranked = sorted(component, key=rank)
        for position, parent_index in enumerate(ranked[:-1]):
            parent = events[parent_index]

            def distance(station):
                lat, lon, _ = geometry[station]
                return math.hypot(
                    111.32 * (lon - parent['lon']) * math.cos(
                        math.radians((lat + parent['lat']) / 2)),
                    111.32 * (lat - parent['lat']))

            observed = {}
            for pick in parent['picks']:
                if pick['s'] > pick['p']:
                    observed.setdefault(station_id(pick['sta']), []).append(
                        (pick['p'], pick['s']))
            distances = [distance(s) for s in observed if s in geometry]
            if not distances:
                continue
            low, high = min(distances), max(distances)
            intervals = {}
            for child_index in ranked[position + 1:]:
                child = events[child_index]
                failed = []
                for row, pick in enumerate(child['picks']):
                    station = station_id(pick['sta'])
                    if station not in intervals:
                        reference = []
                        if station in geometry and low <= distance(station) <= high:
                            reference = observed.get(station, [])
                            if not reference:
                                hypo = math.hypot(distance(station),
                                    parent['depth'] + geometry[station][2] / 1000)
                                reference = [(
                                    parent['time'] + timedelta(seconds=hypo / vp),
                                    parent['time'] + timedelta(seconds=hypo / vs))]
                        intervals[station] = reference
                    duration = (pick['s'] - pick['p']).total_seconds()
                    if duration <= 0:
                        continue
                    if any(max(0, (min(pick['s'], end) - max(pick['p'], start))
                               .total_seconds()) >= overlap_fraction * duration
                           for start, end in intervals[station]):
                        failed.append('{}:{}'.format(row, station))
                if child['picks'] and len(failed) >= failed_fraction * len(child['picks']):
                    rejected.add(child_index)
                    edges.append({
                        'component': component_id,
                        'parent_index': parent_index, 'child_index': child_index,
                        'parent_ot': parent['time'].isoformat(),
                        'parent_lat': parent['lat'], 'parent_lon': parent['lon'],
                        'parent_mag': parent['mag'],
                        'child_ot': child['time'].isoformat(),
                        'child_lat': child['lat'], 'child_lon': child['lon'],
                        'child_mag': child['mag'],
                        'failed_picks': len(failed), 'total_picks': len(child['picks']),
                        'failed_rows': '|'.join(failed),
                    })
    return [e for i, e in enumerate(events) if i not in rejected], edges


def filter_final_events(events, cfg, diagnostic_path):
    """Apply the configured final-stage filter and atomically record its DAG."""
    if not getattr(cfg, 'enable_overlap_duplicate_removal', False):
        return events, 0
    stations = getattr(cfg, '_overlap_duplicate_stations', None)
    if stations is None:
        raise ValueError('overlap duplicate removal requires final-stage station geometry')
    survivors, edges = suppress_overlap_duplicates(
        events, stations,
        origin_tol=float(getattr(cfg, 'overlap_duplicate_origin_time_tol_sec', 10)),
        overlap_fraction=float(getattr(cfg, 'overlap_duplicate_pick_fraction', 0.5)),
        failed_fraction=float(getattr(cfg, 'overlap_duplicate_event_fraction', 0.5)),
        vp=float(cfg.vp), vs=float(cfg.vs))
    path = Path(diagnostic_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    fields = ['component', 'parent_index', 'child_index', 'parent_ot',
              'parent_lat', 'parent_lon', 'parent_mag', 'child_ot', 'child_lat',
              'child_lon', 'child_mag', 'failed_picks', 'total_picks', 'failed_rows']
    with partial.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edges)
    partial.replace(path)
    return survivors, len(events) - len(survivors)
