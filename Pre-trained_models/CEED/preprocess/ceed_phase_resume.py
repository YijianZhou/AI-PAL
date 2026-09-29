"""Per-HDF5 extraction checkpoints and deterministic final output assembly."""
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil


def fingerprint(path, version):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(source=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                version=version)


def part_paths(root, path):
    key = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:24]
    return [Path(root) / (key + suffix) for suffix in ('.pha', '.stations.csv', '.done.json')]


def cached(root, path, signature):
    phase, stations, done = part_paths(root, path)
    try:
        record = json.loads(done.read_text(encoding='utf-8'))
        if record['signature'] != signature:
            return None
        if record['phase_bytes'] != phase.stat().st_size or record['station_bytes'] != stations.stat().st_size:
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def extract_one(task):
    path, root, signature, extract, progress = task
    phase, stations, done = part_paths(root, path)
    phase_tmp, station_tmp = Path(str(phase) + '.tmp'), Path(str(stations) + '.tmp')
    try:
        done.unlink(missing_ok=True)
        counts = extract([Path(path)], str(phase_tmp), str(station_tmp), progress)
        if fingerprint(path, signature['version']) != signature:
            raise RuntimeError('Source HDF5 changed during extraction')
        record = dict(signature=signature, counts=dict(counts),
                      phase_bytes=phase_tmp.stat().st_size,
                      station_bytes=station_tmp.stat().st_size)
        os.replace(phase_tmp, phase)
        os.replace(station_tmp, stations)
        marker_tmp = Path(str(done) + '.tmp')
        marker_tmp.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        os.replace(marker_tmp, done)  # Commit only after both closed outputs exist.
        return str(path), record, None
    except Exception as exc:
        import traceback
        return str(path), None, '{}\n{}'.format(exc, traceback.format_exc())


@contextmanager
def exclusive_run(root):
    # OS locks release after a crash, unlike persistent "in progress" marker files.
    with (root / '.extract.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def extract_resumable(files, phase_out, station_out, part_dir, extract, *,
                      workers=4, resume=True, progress_every=10000,
                      version='ceed-phase-v1'):
    if workers < 1:
        raise ValueError('NUM_WORKERS must be >= 1')
    files = [Path(path).resolve() for path in files]
    if not files or len(set(files)) != len(files):
        raise ValueError('Provide a nonempty list of distinct HDF5 files')
    root = Path(part_dir)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        tasks, records = [], {}
        for path in files:
            signature = fingerprint(path, version)
            record = cached(root, path, signature) if resume else None
            if record is None:
                tasks.append((str(path), str(root), signature, extract, progress_every))
            else:
                records[str(path)] = record
                print('[resume] {}'.format(path), flush=True)
        print('[extract] {} HDF5 files | {} reused | {} pending | {} workers'.format(
            len(files), len(records), len(tasks), min(workers, len(tasks))), flush=True)
        failures = {}
        def collect(result):
            path, record, error = result
            if error:
                failures[path] = error
                print('[failed] {}: {}'.format(path, error.splitlines()[0]), flush=True)
            else:
                records[path] = record
            print('[progress] HDF5 {}/{} finished | {} failed'.format(
                len(records), len(files), len(failures)), flush=True)
        if workers == 1:
            for task in tasks:
                collect(extract_one(task))
        elif tasks:
            with mp.get_context('spawn').Pool(min(workers, len(tasks))) as pool:
                for result in pool.imap_unordered(extract_one, tasks, chunksize=1):
                    collect(result)
        failure_file = root / 'failures.json'
        failure_file.write_text(json.dumps(failures, indent=2) + '\n', encoding='utf-8')
        if failures:
            raise RuntimeError('{} HDF5 files failed; see {}. Completed parts are reusable; final outputs were not replaced.'.format(len(failures), failure_file))

        phase_out = Path(phase_out)
        station_out = Path(station_out) if station_out else None
        phase_out.parent.mkdir(parents=True, exist_ok=True)
        phase_tmp = Path(str(phase_out) + '.tmp')
        totals, station_totals = Counter(), {}
        print('[merge] assembling final phase file from {} completed parts'.format(len(files)), flush=True)
        with phase_tmp.open('wb') as dst:
            for path in files:  # Input order, not worker completion order.
                if fingerprint(path, version) != records[str(path)]['signature']:
                    raise RuntimeError('Source changed before final merge: ' + str(path))
                phase, stations, _ = part_paths(root, path)
                with phase.open('rb') as src:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                totals.update(records[str(path)]['counts'])
                with stations.open(newline='', encoding='utf-8') as fp:
                    for row in csv.DictReader(fp):
                        stats = station_totals.setdefault(row['station_key'], Counter())
                        stats.update({key: int(row[key]) for key in ('pick_rows', 'p_rows', 's_rows', 'ps_rows')})
        if station_out:
            station_out.parent.mkdir(parents=True, exist_ok=True)
            station_tmp = Path(str(station_out) + '.tmp')
            with station_tmp.open('w', newline='', encoding='utf-8') as fp:
                writer = csv.writer(fp, lineterminator='\n')
                fields = ['pick_rows', 'p_rows', 's_rows', 'ps_rows']
                writer.writerow(['station_key'] + fields)
                for key, stats in sorted(station_totals.items(), key=lambda item: (-item[1]['pick_rows'], item[0])):
                    writer.writerow([key] + [stats[field] for field in fields])
            os.replace(station_tmp, station_out)
        os.replace(phase_tmp, phase_out)
        return totals
