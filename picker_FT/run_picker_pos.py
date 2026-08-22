"""Run Frame Transformer positive-window picker on fixed NPY shard datasets."""
import argparse
import json
import os
import re
import time

import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message="Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from picker_pos import FTPositivePicker, configure_positive_picker, rows_to_csv


class NpyShardEventDataset(Dataset):
  def __init__(self, shard_index, sample_index=None):
    self.shard_index = shard_index
    self.rows = np.load(shard_index, allow_pickle=False)
    if self.rows.ndim != 2 or self.rows.shape[1] < 2:
        raise ValueError('Expected shard index with shape (n_shards, 2): {}'.format(shard_index))
    self.sample_index = sample_index
    self.npy_root = os.path.dirname(shard_index)
    self.index_parts_dir = os.path.join(self.npy_root, 'index_parts')

  def __len__(self):
    return self.rows.shape[0]

  def __getitem__(self, index):
    shard_path = str(self.rows[index, 0])
    count = int(self.rows[index, 1])
    shard = np.asarray(np.load(shard_path, mmap_mode='r')[:count], dtype=np.float32)
    meta = self.read_meta_for_shard(shard_path, count)
    return shard, meta

  def read_meta_for_shard(self, shard_path, count):
    part_path = self.index_part_path(shard_path)
    if part_path and os.path.exists(part_path):
        df = pd.read_csv(part_path)
    elif self.sample_index and os.path.exists(self.sample_index):
        df = self.read_meta_from_merged_index(shard_path)
    else:
        df = pd.DataFrame({'row_in_shard': np.arange(count, dtype=np.int64)})
    if len(df) > count:
        df = df.iloc[:count].copy()
    elif len(df) < count:
        pad = pd.DataFrame({'row_in_shard': np.arange(len(df), count, dtype=np.int64)})
        df = pd.concat([df, pad], ignore_index=True)
    # Metadata part files may contain stale paths after a dataset is moved.
    # The shard currently being read is the authoritative sample identity.
    df['shard_path'] = shard_path
    if 'row_in_shard' not in df.columns:
        df['row_in_shard'] = np.arange(len(df), dtype=np.int64)
    return df.to_dict(orient='records')

  def index_part_path(self, shard_path):
    base = os.path.basename(shard_path)
    m_ceed = re.search(r'CEED_file_(\d+)_shard_(\d+)\.npy$', base)
    if m_ceed:
        file_id, shard_id = (int(value) for value in m_ceed.groups())
        candidates = [
            os.path.join(self.index_parts_dir, 'part_{}_{}.csv'.format(file_id, shard_id)),
            os.path.join(self.index_parts_dir, 'part_%05d_%06d.csv' % (file_id, shard_id)),
        ]
    else:
        m = re.search(r'_shard_(\d+)\.npy$', base)
        if not m:
            return None
        shard_id = m.group(1)
        candidates = [
            os.path.join(self.index_parts_dir, 'part_{}.csv'.format(shard_id)),
            os.path.join(self.index_parts_dir, 'part_%06d.csv' % int(shard_id)),
        ]
        candidates.extend(sorted(glob_join(self.index_parts_dir, 'part_*_{}.csv'.format(shard_id))))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

  def read_meta_from_merged_index(self, shard_path):
    chunks = []
    for chunk in pd.read_csv(self.sample_index, chunksize=200000):
        if 'shard_path' not in chunk.columns:
            continue
        hit = chunk[chunk['shard_path'] == shard_path]
        if not hit.empty:
            chunks.append(hit.copy())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def glob_join(root, pattern):
    import glob
    return glob.glob(os.path.join(root, pattern))


def collate_identity(item):
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--shard_index', type=str, required=True)
    parser.add_argument('--sample_index', type=str, default='')
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_idx', type=int, default=-1)
    parser.add_argument('--pos_start_min', type=float, required=True)
    parser.add_argument('--pos_start_max', type=float, required=True)
    parser.add_argument('--pos_num_repeat', type=int, required=True)
    parser.add_argument('--pos_batch_size', type=int, required=True)
    parser.add_argument('--pos_random_seed', type=int, required=True)
    parser.add_argument('--pos_min_cluster_size', type=int, required=True)
    args = parser.parse_args()
    configure_positive_picker(
        [args.pos_start_min, args.pos_start_max],
        args.pos_num_repeat,
        args.pos_batch_size,
        args.pos_random_seed,
        args.pos_min_cluster_size,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)
        torch.backends.cudnn.benchmark = True

    sample_index = args.sample_index if args.sample_index else os.path.join(os.path.dirname(args.shard_index), 'sample_index.csv')
    dataset = NpyShardEventDataset(args.shard_index, sample_index)
    loader_kwargs = dict(batch_size=None, num_workers=args.num_workers, collate_fn=collate_identity, pin_memory=False)
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    loader = DataLoader(dataset, **loader_kwargs)

    picker = FTPositivePicker(args.ckpt_dir, args.ckpt_idx, args.gpu_idx)
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    partial_file = args.out_file + '.partial'
    summary_file = args.out_file + '.summary.json'
    summary_partial = summary_file + '.partial'
    expected_shards = len(dataset)
    expected_samples = int(np.asarray(dataset.rows[:, 1], dtype=np.int64).sum())
    print(
        'expected | shards {:,} | samples {:,} | temporary output {}'.format(
            expected_shards, expected_samples, partial_file
        ),
        flush=True,
    )

    t0 = time.time()
    num_shards = 0
    num_samples = 0
    num_picks = 0
    try:
        with open(partial_file, 'w') as fout:
            rows_to_csv([], fout, write_header=True)
            for shard, meta in loader:
                rows = picker.pick_shard(shard, meta)
                rows_to_csv(rows, fout, write_header=False)
                num_shards += 1
                num_samples += int(shard.shape[0])
                num_picks += len(rows)
                if num_shards % 50 == 0:
                    fout.flush()
                    dt = max(time.time() - t0, 1e-6)
                    print('shards {:,}/{:,} | samples {:,}/{:,} | picks {:,} | {:.1f} samples/s'.format(
                        num_shards, expected_shards, num_samples, expected_samples,
                        num_picks, num_samples / dt
                    ), flush=True)

        if num_shards != expected_shards or num_samples != expected_samples:
            raise RuntimeError(
                'Incomplete inference: processed {}/{} shards and {}/{} samples; '
                'partial output retained at {}'.format(
                    num_shards, expected_shards, num_samples, expected_samples, partial_file
                )
            )

        elapsed = time.time() - t0
        summary = {
            'complete': True,
            'shard_index': os.path.abspath(args.shard_index),
            'sample_index': os.path.abspath(sample_index),
            'num_shards': num_shards,
            'num_samples': num_samples,
            'num_pick_rows': num_picks,
            'elapsed_sec': elapsed,
        }
        with open(summary_partial, 'w') as fout:
            json.dump(summary, fout, indent=2, sort_keys=True)
            fout.write('\n')
        os.replace(partial_file, args.out_file)
        os.replace(summary_partial, summary_file)
    except BaseException:
        print('inference did not complete; final output was not replaced', flush=True)
        print('partial output: {}'.format(partial_file), flush=True)
        raise

    print('done | shards {:,} | samples {:,} | picks {:,} | {:.1f}s'.format(
        num_shards, num_samples, num_picks, elapsed
    ), flush=True)
    print('output: {}'.format(args.out_file), flush=True)
    print('summary: {}'.format(summary_file), flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()