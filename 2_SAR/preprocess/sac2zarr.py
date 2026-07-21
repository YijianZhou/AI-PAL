"""Make SAR Zarr dataset with raw waveform SAC files."""
import os
import argparse
import time
import zarr
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader
from dataset_sac import Sequences
import config
import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.rnn_num_steps
win_len = int(cfg.win_len * samp_rate)


def write_sequence(zarr_dset, data_loader, chunk_size=256):
    num_samples = len(data_loader.dataset)
    data_shape = (num_samples, num_chn, win_len)
    target_shape = (num_samples, num_steps)
    sample_chunk = max(1, min(chunk_size, num_samples))
    data_out = os.path.join(out_path, zarr_dset+'_data')
    target_out = os.path.join(out_path, zarr_dset+'_target_sar')
    print('writing %s & %s' % (data_out, target_out), flush=True)
    z_data = zarr.open(data_out, mode='w', shape=data_shape, chunks=(sample_chunk, num_chn, win_len), dtype=np.float32)
    z_target = zarr.open(target_out, mode='w', shape=target_shape, chunks=(sample_chunk, num_steps), dtype=np.int32)
    cursor = 0
    t0 = time.time()
    last_t = t0
    for data, target in data_loader:
        load_t = time.time()
        data_np = data.detach().cpu().numpy() if hasattr(data, 'detach') else np.asarray(data)
        target_np = target.detach().cpu().numpy() if hasattr(target, 'detach') else np.asarray(target)
        data_np = data_np.astype(np.float32, copy=False)
        target_np = target_np.astype(np.int32, copy=False)
        if data_np.ndim == 2:
            data_np = data_np[None, ...]
        if target_np.ndim == 1:
            target_np = target_np[None, ...]
        next_cursor = cursor + data_np.shape[0]
        z_data[cursor:next_cursor] = data_np
        z_target[cursor:next_cursor] = target_np
        write_t = time.time()
        cursor = next_cursor
        if cursor == num_samples or cursor % args.log_interval < data_np.shape[0]:
            print('done / total = %d / %d | %.1f samp/s | load %.2fs write %.2fs recent %.1fs' % (
                cursor, num_samples, cursor / max(write_t - t0, 1e-6), load_t - last_t, write_t - load_t, write_t - last_t), flush=True)
            last_t = write_t


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--sac_root', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=100000)
    args = parser.parse_args()
    out_path = args.out_path
    train_pos = os.path.join(args.sac_root, 'train_pos.npy')
    valid_pos = os.path.join(args.sac_root, 'valid_pos.npy')
    train_neg = os.path.join(args.sac_root, 'train_neg.npy')
    valid_neg = os.path.join(args.sac_root, 'valid_neg.npy')
    train_pos_loader = DataLoader(Sequences(train_pos, True), batch_size=args.chunk_size, shuffle=False, num_workers=args.num_workers)
    valid_pos_loader = DataLoader(Sequences(valid_pos, True), batch_size=args.chunk_size, shuffle=False, num_workers=args.num_workers)
    train_neg_loader = DataLoader(Sequences(train_neg, False), batch_size=args.chunk_size, shuffle=False, num_workers=args.num_workers)
    valid_neg_loader = DataLoader(Sequences(valid_neg, False), batch_size=args.chunk_size, shuffle=False, num_workers=args.num_workers)
    write_sequence('train/positive', train_pos_loader, args.chunk_size)
    write_sequence('valid/positive', valid_pos_loader, args.chunk_size)
    write_sequence('train/negative', train_neg_loader, args.chunk_size)
    write_sequence('valid/negative', valid_neg_loader, args.chunk_size)