"""Offline event timing sidecars and readable daily reports (no plotting)."""
import csv
import json
from pathlib import Path


def write_event_timing(root, key, timing):
    path = Path(root) / 'timing.json'
    partial = path.with_suffix('.json.partial')
    partial.write_text(json.dumps(dict(key=key, **timing), indent=2), encoding='utf-8')
    partial.replace(path)


def export_daily_timing(cache, output_dir, day, finalization):
    """Missing legacy timings remain missing, never represented as zero seconds."""
    rows = []
    for index, (root, key) in enumerate(cache):
        row = dict(event_index=index, timing_available=False)
        try:
            timing = json.loads((Path(root) / 'timing.json').read_text())
            if timing.pop('key') == key:
                row.update(timing, timing_available=True)
        except (OSError, ValueError, KeyError):
            pass
        rows.append(row)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ['event_index', 'timing_available'] + sorted(
        set().union(*(set(row) for row in rows)) - {'event_index', 'timing_available'})
    csv_path = output_dir / ('events_{}.csv'.format(day))
    partial = csv_path.with_suffix('.csv.partial')
    with partial.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(csv_path)
    totals = {name: sum(row.get(name, 0) for row in rows)
              for name in columns if name.endswith('_sec')}
    result = dict(
        event_count=len(rows),
        timed_event_count=sum(row['timing_available'] for row in rows),
        summed_event_seconds=totals,
        finalization=finalization,
        note='Event times may overlap and may come from resumed runs. Sums are not day wall time.',
    )
    path = output_dir / ('summary_{}.json'.format(day))
    partial = path.with_suffix('.json.partial')
    partial.write_text(json.dumps(result, indent=2), encoding='utf-8')
    partial.replace(path)
    return result
