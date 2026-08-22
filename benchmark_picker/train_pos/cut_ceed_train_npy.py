"""Cut CEED HDF5 station waveforms into augmented 25 s NPY shards.

This direct HDF5 -> NPY-shard path avoids creating millions of intermediate
SAC channel files while retaining the same waveform preprocessing operations.

Output layout:
    out_root/train/group_00000_shard_000000.npy
    out_root/valid/group_00000_shard_000000.npy
    out_root/train_pos.npy
    out_root/valid_pos.npy

Each shard is a float32 array with shape (n_samples, 3, win_len + 2):
    shard[:, channel, 0] = P arrival time relative to window start
    shard[:, channel, 1] = S arrival time relative to window start
    shard[:, channel, 2:] = waveform samples

The train_pos.npy and valid_pos.npy files are small shard indexes with rows:
    shard_path, sample_count
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import multiprocessing as mp

import h5py
import numpy as np

import ceed_waveform as sac

DEFAULT_OUT_ROOT = Path('/nas/zhouyj/CEED_train_npy')

# =============================================================================
# USER SETTINGS
# =============================================================================
PHASE_FILE = 'output/ceed_phase_train_augmented.pha'
CEED_ROOT = sac.DEFAULT_CEED_ROOT
NC_DIR = None
SC_DIR = None
OUT_ROOT = DEFAULT_OUT_ROOT
SUMMARY_OUT = 'output/ceed_cut_train_npy_summary.csv'
WINDOW_LENGTH = 25.0
SAMPLE_RATE = 100.0
PHASE_MARGIN = 0.2
FREQMIN = 1.0
FREQMAX = 20.0
FILTER_CORNERS = 4
TAPER_MAX_PERCENTAGE = 0.05
TAPER_MAX_LENGTH = 5.0
EVENT_TIME_TOLERANCE_SEC = 0.01
RANDOM_SEED = 20250701
PROGRESS_EVERY = 10000
NUM_WORKERS = 10
SHARD_SIZE = 1024
INTEGRATE_ACCELERATION = True
VERBOSE_ERRORS = False


def make_npy_sample(stream, sample, start_time, args):
    p_rel = float(sac.utc(sample['tp']) - start_time)
    s_rel = float(sac.utc(sample['ts']) - start_time)
    win_len = int(round(args.window_length * args.sample_rate))
    out = np.zeros((3, win_len + 2), dtype=np.float32)
    out[:, 0] = p_rel
    out[:, 1] = s_rel
    for ii, tr in enumerate(stream[:3]):
        data = np.asarray(tr.data, dtype=np.float32)
        npts = min(data.size, win_len)
        out[ii, 2:2+npts] = data[:npts]
    return out


def write_shard(out_root, split, group_index, shard_index, samples):
    split_dir = Path(out_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard = np.stack(samples, axis=0).astype(np.float32, copy=False)
    shard_path = split_dir / ('group_%05d_shard_%06d.npy' % (group_index, shard_index))
    np.save(shard_path, shard)
    return str(shard_path), shard.shape[0]


def flush_split(out_root, split, group_index, shard_state, force=False):
    buffers = shard_state[split]['buffers']
    rows = shard_state[split]['rows']
    shard_index = shard_state[split]['next_idx']
    while len(buffers) >= shard_state['shard_size'] or (force and buffers):
        nwrite = min(len(buffers), shard_state['shard_size'])
        path, count = write_shard(out_root, split, group_index, shard_index, buffers[:nwrite])
        rows.append((path, str(count)))
        del buffers[:nwrite]
        shard_index += 1
    shard_state[split]['next_idx'] = shard_index


def process_sample(dataset, sample, args, rng):
    counts = Counter()
    sample_arrays = []
    try:
        stream = sac.make_stream(dataset, sample, args)
        if sample.get('integrated_acceleration'):
            counts['acceleration_records_integrated'] += 1
        if abs(stream[0].stats.sampling_rate - args.sample_rate) > 1e-6:
            stream.resample(args.sample_rate)
        stream = sac.preprocess_stream(stream, args)
        window_range = sac.valid_window_start_range(stream, sample['tp'], sample['ts'], args)
        if window_range is None:
            counts['skipped_no_valid_window'] += 1
            return counts, sample_arrays
        min_start, max_start = window_range
        span = max_start - min_start
        for aug_idx in range(sample['num_aug']):
            offset = rng.random() * span if span > 0 else 0.0
            start_time = min_start + offset
            cut = sac.cut_and_normalize(stream, start_time, args)
            if len(cut) != 3:
                counts['skipped_bad_cut_channel_count'] += 1
                continue
            sample_arrays.append(make_npy_sample(cut, sample, start_time, args))
            counts['augmented_samples_written'] += 1
    except Exception as exc:
        counts['skipped_exception'] += 1
        if args.verbose_errors:
            print(f"sample line {sample.get('line_number')} failed: {exc}", flush=True)
    return counts, sample_arrays


def process_h5_group(task):
    group_index, h5_path, samples, args = task
    counts = Counter()
    shard_state = {
        'shard_size': args.shard_size,
        'train': {'buffers': [], 'rows': [], 'next_idx': 0},
        'valid': {'buffers': [], 'rows': [], 'next_idx': 0},
    }
    rng = np.random.default_rng(args.random_seed + group_index)
    print(f"reading {h5_path}", flush=True)
    with h5py.File(h5_path, 'r') as h5:
        by_event = defaultdict(list)
        for sample in samples:
            by_event[sample['h5_event_id']].append(sample)
        done = 0
        for event_id in sorted(by_event):
            if event_id not in h5:
                counts['missing_event_in_h5'] += len(by_event[event_id])
                continue
            event = h5[event_id]
            for sample in by_event[event_id]:
                station_key = sample['station_key']
                if station_key not in event:
                    counts['missing_station_dataset'] += 1
                    continue
                sample_counts, sample_arrays = process_sample(event[station_key], sample, args, rng)
                counts.update(sample_counts)
                if sample_arrays:
                    split = sample['split']
                    shard_state[split]['buffers'].extend(sample_arrays)
                    flush_split(args.out_root, split, group_index, shard_state, force=False)
                done += 1
                if args.progress_every and done % args.progress_every == 0:
                    print(f"{h5_path}: processed {done:,}/{len(samples):,}", flush=True)
    for split in ('train', 'valid'):
        flush_split(args.out_root, split, group_index, shard_state, force=True)
        counts[f'{split}_npy_shards_written'] += len(shard_state[split]['rows'])
    counts['phase_samples_processed'] += len(samples)
    return counts, shard_state['train']['rows'], shard_state['valid']['rows']


def process_samples(samples, args):
    counts = Counter()
    train_rows = []
    valid_rows = []
    by_path = defaultdict(list)
    for sample in samples:
        by_path[str(Path(sample['h5_path']))].append(sample)
    tasks = [(idx, h5_path, by_path[h5_path], args) for idx, h5_path in enumerate(sorted(by_path))]
    if args.num_workers <= 1 or len(tasks) == 1:
        results = [process_h5_group(task) for task in tasks]
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(pool.imap_unordered(process_h5_group, tasks))
    for group_counts, group_train_rows, group_valid_rows in results:
        counts.update(group_counts)
        train_rows.extend(group_train_rows)
        valid_rows.extend(group_valid_rows)
    train_rows.sort(key=lambda item: item[0])
    valid_rows.sort(key=lambda item: item[0])
    return counts, train_rows, valid_rows


def save_shard_indexes(out_root, train_rows, valid_rows):
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / 'train_pos.npy', np.asarray(train_rows, dtype=str))
    np.save(root / 'valid_pos.npy', np.asarray(valid_rows, dtype=str))


def write_summary(path, phase_counts, resolve_counts, process_counts, args, n_samples, n_resolved):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(['parameter', 'value'])
        writer.writerow(['phase_file', args.phase_file])
        writer.writerow(['ceed_root', args.ceed_root])
        writer.writerow(['out_root', args.out_root])
        writer.writerow(['window_length', args.window_length])
        writer.writerow(['sample_rate', args.sample_rate])
        writer.writerow(['freqmin', args.freqmin])
        writer.writerow(['freqmax', args.freqmax])
        writer.writerow(['phase_margin', args.phase_margin])
        writer.writerow(['random_seed', args.random_seed])
        writer.writerow(['num_workers', args.num_workers])
        writer.writerow(['shard_size', args.shard_size])
        writer.writerow(['samples_read', n_samples])
        writer.writerow(['samples_resolved', n_resolved])
        for prefix, counts in [('phase', phase_counts), ('resolve', resolve_counts), ('process', process_counts)]:
            for key in sorted(counts):
                writer.writerow([f'{prefix}_{key}', counts[key]])


def main():
    from types import SimpleNamespace
    args = SimpleNamespace(
        phase_file=PHASE_FILE, ceed_root=str(CEED_ROOT), nc_dir=NC_DIR, sc_dir=SC_DIR,
        out_root=str(OUT_ROOT), summary_out=SUMMARY_OUT, window_length=WINDOW_LENGTH,
        sample_rate=SAMPLE_RATE, phase_margin=PHASE_MARGIN, freqmin=FREQMIN,
        freqmax=FREQMAX, filter_corners=FILTER_CORNERS,
        taper_max_percentage=TAPER_MAX_PERCENTAGE, taper_max_length=TAPER_MAX_LENGTH,
        event_time_tolerance_sec=EVENT_TIME_TOLERANCE_SEC, random_seed=RANDOM_SEED,
        progress_every=PROGRESS_EVERY, num_workers=NUM_WORKERS, shard_size=SHARD_SIZE,
        integrate_acceleration=INTEGRATE_ACCELERATION, verbose_errors=VERBOSE_ERRORS,
    )
    print(f"reading phase file: {args.phase_file}", flush=True)
    samples, phase_counts = sac.read_training_phase(args.phase_file)
    print(f"phase rows loaded: {len(samples):,}", flush=True)
    roots = sac.ceed_h5_roots(args)
    print("CEED HDF5 roots: " + ", ".join(str(root) for root in roots), flush=True)
    resolved, resolve_counts = sac.resolve_samples(samples, roots, args.event_time_tolerance_sec)
    print(f"resolved phase rows: {len(resolved):,}", flush=True)
    process_counts, train_rows, valid_rows = process_samples(resolved, args)
    save_shard_indexes(args.out_root, train_rows, valid_rows)
    write_summary(args.summary_out, phase_counts, resolve_counts, process_counts, args, len(samples), len(resolved))
    print(f"phase file: {args.phase_file}")
    print(f"samples read: {len(samples)}")
    print(f"samples resolved: {len(resolved)}")
    print(f"augmented samples written: {process_counts.get('augmented_samples_written', 0)}")
    print(f"train NPY shards written: {process_counts.get('train_npy_shards_written', 0)}")
    print(f"valid NPY shards written: {process_counts.get('valid_npy_shards_written', 0)}")
    print(f"train shard index: {Path(args.out_root) / 'train_pos.npy'}")
    print(f"valid shard index: {Path(args.out_root) / 'valid_pos.npy'}")
    print(f"summary: {args.summary_out}")


if __name__ == '__main__':
    main()
