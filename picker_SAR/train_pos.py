"""Train SAR with positive earthquake-window samples only."""
import os, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import torch.multiprocessing as mp
from dataset_pos import PositiveOnly
from models import SAR
import config
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

class ChunkBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        self.chunk_size = max(1, int(dataset.chunk_size()))

    def __iter__(self):
        chunk_starts = list(range(0, self.num_samples, self.chunk_size))
        perm = torch.randperm(len(chunk_starts)).tolist()
        for chunk_idx in perm:
            start = chunk_starts[chunk_idx]
            end = min(start + self.chunk_size, self.num_samples)
            local = torch.randperm(end - start).tolist()
            batch = []
            for off in local:
                batch.append(start + off)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch and not self.drop_last:
                yield batch

    def __len__(self):
        total_batches = 0
        for start in range(0, self.num_samples, self.chunk_size):
            chunk_samples = min(self.chunk_size, self.num_samples - start)
            if self.drop_last:
                total_batches += chunk_samples // self.batch_size
            else:
                total_batches += (
                    chunk_samples + self.batch_size - 1
                ) // self.batch_size
        return total_batches


def make_loader(dataset, batch_size, shuffle, args):
    kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    if shuffle and args.chunk_shuffle:
        return DataLoader(dataset, batch_sampler=ChunkBatchSampler(dataset, batch_size), **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def next_valid_batch(valid_loader, valid_iter):
    try:
        return next(valid_iter), valid_iter
    except StopIteration:
        valid_iter = iter(valid_loader)
        return next(valid_iter), valid_iter


def main():
    torch.backends.cudnn.benchmark = True
    cfg = config.Config()
    train_set = PositiveOnly(args.zarr_path, 'train')
    valid_set = PositiveOnly(args.zarr_path, 'valid')
    train_loader = make_loader(train_set, cfg.batch_size, True, args)
    valid_loader = make_loader(valid_set, cfg.batch_size, False, args)
    valid_iter = iter(valid_loader)
    model = SAR()
    device = torch.device("cuda:%s" % args.gpu_idx if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    num_batch = len(train_loader)
    t = time.time()
    writer = SummaryWriter(log_dir=args.ckpt_dir)
    try:
        for epoch_idx in range(cfg.num_epochs):
          for iter_idx, (data, target) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()
            data = unfold_sar_batch(data, cfg)
            train_acc, train_pick_acc, train_loss = train_step(model, data, target, criterion, optimizer)
            if global_step % cfg.ckpt_step == 0:
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, '%s_%s-%s.ckpt' % (global_step, epoch_idx, iter_idx)))
            if global_step % cfg.summary_step != 0:
                continue
            (data_v, target_v), valid_iter = next_valid_batch(valid_loader, valid_iter)
            data_v = data_v.to(device, non_blocking=True).float()
            target_v = target_v.to(device, non_blocking=True).long()
            data_v = unfold_sar_batch(data_v, cfg)
            valid_acc, valid_pick_acc, valid_loss = valid_step(model, data_v, target_v, criterion)
            print('step {} ({}/{}) | train loss {:.2f} | valid loss {:.2f} | {:.2f}s'.format(
                global_step, iter_idx, epoch_idx, train_loss, valid_loss, time.time()-t))
            print('   frame acc. {:.2f}% {:.2f}% | pick acc. {:.2f}% {:.2f}%'.format(
                100*train_acc, 100*valid_acc, 100*train_pick_acc, 100*valid_pick_acc))
            writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, global_step)
            writer.add_scalars('frame_acc', {'train_frame_acc': 100*train_acc, 'valid_frame_acc': 100*valid_acc}, global_step)
            writer.add_scalars('pick_acc', {'train_pick_acc': 100*train_pick_acc, 'valid_pick_acc': 100*valid_pick_acc}, global_step)
    finally:
        writer.close()

def unfold_sar_batch(data, cfg):
    step_len = int(cfg.rnn_step_len * cfg.samp_rate)
    step_stride = int(cfg.rnn_step_stride * cfg.samp_rate)
    num_steps = cfg.rnn_num_steps
    data_seq = data.unfold(2, step_len, step_stride).permute(0, 2, 1, 3).reshape(data.size(0), -1, step_len * cfg.num_chn)
    if data_seq.size(1) < num_steps:
        pad = data_seq.new_zeros(data_seq.size(0), num_steps - data_seq.size(1), data_seq.size(2))
        data_seq = torch.cat((data_seq, pad), dim=1)
    elif data_seq.size(1) > num_steps:
        data_seq = data_seq[:, 0:num_steps, :]
    return data_seq.contiguous()


def batch_metrics(pred_class, target):
    frame_acc = pred_class.reshape(-1).eq(target.reshape(-1)).sum() / float(target.numel())
    pred_pos = (pred_class == 1).any(dim=1) & (pred_class == 2).any(dim=1)
    pick_acc = pred_pos.sum() / float(pred_pos.numel())
    return frame_acc.item(), pick_acc.item()


def train_step(model, data, target, criterion, optimizer):
    model.train()
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits, 2)
    loss = criterion(pred_logits.reshape(-1, 3), target.reshape(-1))
    frame_acc, pick_acc = batch_metrics(pred_class, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return frame_acc, pick_acc, loss.item()


def valid_step(model, data, target, criterion):
    model.eval()
    with torch.no_grad():
        pred_logits = model(data)
        pred_class = torch.argmax(pred_logits, 2)
        loss = criterion(pred_logits.reshape(-1, 3), target.reshape(-1))
        frame_acc, pick_acc = batch_metrics(pred_class, target)
    return frame_acc, pick_acc, loss.item()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument(
        '--chunk_shuffle',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    main()
