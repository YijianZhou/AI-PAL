"""Final AI-PAL magnitude QC using Local+Global station coverage."""
import csv
import math
from pathlib import Path


def parameters(cfg):
    action = getattr(cfg, 'station_participation_qc_action', 'mark_nan')
    if action not in ('off', 'mark_nan', 'remove'):
        raise ValueError('station_participation_qc_action must be off, mark_nan, or remove')
    mags = tuple(getattr(cfg, 'station_participation_qc_magnitudes', (2.0, 2.5)))
    ratios = tuple(getattr(cfg, 'station_participation_qc_ratios', (0.05, 0.10)))
    if (len(mags) != 2 or len(ratios) != 2
            or not all(math.isfinite(x) for x in mags + ratios)
            or not mags[0] < mags[1] or not 0 < ratios[0] < ratios[1] <= 1):
        raise ValueError('Participation QC requires increasing finite magnitude/ratio pairs, 0 < ratios <= 1')
    return action, mags, ratios


def magnitude_limit(ratio, magnitudes, ratios):
    """Strictly above this boundary is cleaned; ratios >= upper bound pass."""
    if ratio >= ratios[1]:
        return math.inf
    fraction = max(0.0, (ratio - ratios[0]) / (ratios[1] - ratios[0]))
    return magnitudes[0] + fraction * (magnitudes[1] - magnitudes[0])


def evaluate(event, stations, magnitudes, ratios):
    identity = lambda value: '.'.join(str(value).split('.')[:2])
    geometry = {}
    for key, values in stations.items():
        xy = tuple(float(x) for x in values[:2])
        if len(xy) == 2 and all(math.isfinite(x) for x in xy):
            geometry.setdefault(identity(key), xy)
    associated = {identity(p['sta']) for p in event['picks']}
    anchors = {identity(p['sta']) for p in event['picks']
               if 'both_groups' in str(p.get('pick_provenance', '')).split('|')}
    row = dict(origin_time=event['time'].isoformat(), latitude=event['lat'],
               longitude=event['lon'], original_mag=event['mag'],
               radius_km=None, associated_within_radius=None, inventory_within_radius=None,
               ratio=None, magnitude_limit=None, action='unchanged', status='ok')
    if not anchors or not associated <= geometry.keys():
        row['status'] = 'no_both_group_picks' if not anchors else 'missing_station_geometry'
        return row
    distances = {s: math.hypot(111.32 * (lat-event['lat']),
                 111.32 * (lon-event['lon']) * math.cos(math.radians((lat+event['lat'])/2)))
                 for s, (lat, lon) in geometry.items()}
    radius = max(distances[s] for s in anchors)
    numerator = sum(distances[s] <= radius + 1e-9 for s in associated)
    denominator = sum(d <= radius + 1e-9 for d in distances.values())
    ratio = numerator / denominator
    limit = magnitude_limit(ratio, magnitudes, ratios)
    row.update(radius_km=radius, associated_within_radius=numerator,
               inventory_within_radius=denominator, ratio=ratio, magnitude_limit=limit)
    mag = float(event['mag'])
    if not math.isfinite(mag) or mag == -1:
        row['status'] = 'missing_magnitude'
    elif mag > limit:
        row['status'] = 'flagged'
    return row


def filter_events(events, cfg, diagnostic_path):
    action, magnitudes, ratios = parameters(cfg)
    if action == 'off':
        return events
    # Reference catalogs have no Local+Global provenance or radius definition.
    if not any('both_groups' in str(p.get('pick_provenance', '')).split('|')
               for e in events for p in e['picks']):
        return events
    stations = getattr(cfg, '_overlap_duplicate_stations', None)
    if stations is None:
        raise ValueError('Station participation QC requires final-stage station inventory')
    accepted, rows = [], []
    for event in events:
        row = evaluate(event, stations, magnitudes, ratios)
        rows.append(row)
        if row['status'] == 'flagged':
            row['action'] = action
            if action == 'remove':
                continue
            event = dict(event, mag=float('nan'))
        accepted.append(event)
    path = Path(diagnostic_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('w', newline='', encoding='utf-8') as fp:
        fields = list(rows[0]) if rows else ['origin_time', 'status', 'action']
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)
    return accepted
