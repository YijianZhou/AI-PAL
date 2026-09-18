"""Add positive/negative Frame Transformer targets to the shared Zarr."""
import os
import sys
import importlib.util

package_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(package_dir, 'config.py')
spec = importlib.util.spec_from_file_location('config', config_path)
config = importlib.util.module_from_spec(spec)
sys.modules['config'] = config
spec.loader.exec_module(config)

import argparse
import time
import numpy as np
import zarr
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from dataset_npy import NpyFrameShards
import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = int(cfg.win_len * samp_rate)
num_steps = int((cfg.win_len - cfg.ft_frame_length) / cfg.ft_frame_step) + 1


def get_compressor(name):
    if name == 'none':
        return None
    from numcodecs import Blosc
    if name == 'zstd':
        return Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    return Blosc(cname='lz4', clevel=1, shuffle=Blosc.BITSHUFFLE)


def collate_shard(batch):
    return batch[0]


def zarr_exists(path):
    return os.path.isdir(path) or os.path.exists(path)


def make_loader(sample_npy, num_workers, write_batch_size, prefetch_factor, include_data):
    dataset = NpyFrameShards(sample_npy, include_data=include_data)
    kwargs = dict(batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False, collate_fn=collate_shard)
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor)
    return DataLoader(dataset, **kwargs)


def array_is_compatible(path, shape):
    if not zarr_exists(path):
        return False
    arr = zarr.open(path, mode='r')
    return tuple(arr.shape) == tuple(shape)


def write_sequence(zarr_dset, sample_npy, chunk_size, compressor, log_interval, args):
    probe = NpyFrameShards(sample_npy, include_data=False)
    num_samples = probe.num_samples
    data_shape = (num_samples, num_chn, win_len)
    target_shape = (num_samples, num_steps)
    data_out = os.path.join(out_path, zarr_dset+'_data')
    target_out = os.path.join(out_path, zarr_dset+'_target_frame')
    write_data = not array_is_compatible(data_out, data_shape)
    write_target = not array_is_compatible(target_out, target_shape)
    if not write_data and not write_target:
        print('reusing compatible %s and %s' % (data_out, target_out), flush=True)
        return
    print('%s; %s' % (
        'writing %s' % data_out if write_data else 'reusing %s' % data_out,
        'writing %s' % target_out if write_target else 'reusing %s' % target_out,
    ), flush=True)

    sample_chunk = max(1, min(chunk_size, num_samples))
    if write_data:
        z_data = zarr.open(data_out, mode='w', shape=data_shape, chunks=(sample_chunk, num_chn, win_len), dtype=np.float32, compressor=compressor)
    else:
        z_data = None
    z_target = zarr.open(target_out, mode='w', shape=target_shape, chunks=(sample_chunk, num_steps), dtype=np.int32, compressor=compressor) if write_target else None

    data_loader = make_loader(sample_npy, args.num_workers, args.write_batch_size, args.prefetch_factor, include_data=write_data)
    cursor = 0
    t0 = time.time()
    last_t = t0
    for data, target in data_loader:
        load_t = time.time()
        target_np = np.asarray(target, dtype=np.int32)
        next_cursor = cursor + target_np.shape[0]
        if write_data:
            data_np = np.asarray(data, dtype=np.float32)
            z_data[cursor:next_cursor] = data_np
        if write_target:
            z_target[cursor:next_cursor] = target_np
        write_t = time.time()
        cursor = next_cursor
        if cursor == num_samples or cursor % log_interval < target_np.shape[0]:
            print('done / total = %d / %d | %.1f samp/s | load %.2fs write %.2fs recent %.1fs' % (
                cursor, num_samples, cursor / max(write_t - t0, 1e-6), load_t - last_t, write_t - load_t, write_t - last_t), flush=True)
            last_t = write_t


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_root', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--write_batch_size', type=int, default=128, help='Kept for CLI compatibility; shard size controls actual batch size.')
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--prefetch_factor', type=int, default=1)
    parser.add_argument('--compressor', type=str, default='lz4', choices=['none', 'lz4', 'zstd'])
    parser.add_argument('--log_interval', type=int, default=100000)
    parser.add_argument('--positive_only', action='store_true')
    args = parser.parse_args()
    out_path = args.out_path
    compressor = get_compressor(args.compressor)
    write_sequence('train/positive', os.path.join(args.npy_root, 'train_pos.npy'), args.chunk_size, compressor, args.log_interval, args)
    write_sequence('valid/positive', os.path.join(args.npy_root, 'valid_pos.npy'), args.chunk_size, compressor, args.log_interval, args)
    if not args.positive_only:
        write_sequence('train/negative', os.path.join(args.npy_root, 'train_neg.npy'), args.chunk_size, compressor, args.log_interval, args)
        write_sequence('valid/negative', os.path.join(args.npy_root, 'valid_neg.npy'), args.chunk_size, compressor, args.log_interval, args)
