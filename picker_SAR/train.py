"""Train SAR with positive-only or positive-and-negative samples."""
import os, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import torch.multiprocessing as mp
from dataset import Positive_Negative, PositiveOnly
from models import SAR
import config
from tensorboardX import SummaryWriter
from training_monitor import TrainingMonitor
from torch_backends import configure_torch_backends

# Also runs when spawned DataLoader workers import this module.
configure_torch_backends(config.Config())
import warnings
warnings.filterwarnings("ignore")


def prune_numbered_checkpoints(ckpt_dir, keep):
    keep = int(keep)
    if keep < 1:
        raise ValueError('max_checkpoints must be at least 1')
    checkpoints = []
    for name in os.listdir(ckpt_dir):
        step = name.split('_', 1)[0]
        if step.isdigit() and name.endswith('.ckpt'):
            checkpoints.append((int(step), name))
    for _, name in sorted(checkpoints)[:-keep]:
        os.remove(os.path.join(ckpt_dir, name))


class ChunkBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.num_samples = len(dataset)
        self.chunk_size = max(1, int(dataset.chunk_size()))

    def __iter__(self):
        chunk_starts = list(range(0, self.num_samples, self.chunk_size))
        for chunk_idx in torch.randperm(len(chunk_starts)).tolist():
            start = chunk_starts[chunk_idx]
            end = min(start + self.chunk_size, self.num_samples)
            batch = []
            for offset in torch.randperm(end - start).tolist():
                batch.append(start + offset)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch and not self.drop_last:
                yield batch

    def __len__(self):
        total = 0
        for start in range(0, self.num_samples, self.chunk_size):
            count = min(self.chunk_size, self.num_samples - start)
            total += count // self.batch_size if self.drop_last else (
                count + self.batch_size - 1
            ) // self.batch_size
        return total


def make_loader(dataset, batch_size, shuffle):
    kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    if shuffle and args.chunk_shuffle:
        return DataLoader(
            dataset,
            batch_sampler=ChunkBatchSampler(dataset, batch_size),
            **kwargs
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def main():
    torch.backends.cudnn.benchmark = True
    cfg = config.Config()
    lr = cfg.lr
    num_epochs = cfg.num_epochs
    summary_step = cfg.summary_step
    valid_step_interval = cfg.valid_step
    max_checkpoints = int(getattr(cfg, 'max_checkpoints', 20))
    batch_size = cfg.batch_size
    dataset_class = PositiveOnly if args.positive_only else Positive_Negative
    train_set = dataset_class(args.zarr_path, 'train')
    valid_set = dataset_class(args.zarr_path, 'valid')
    train_loader = make_loader(train_set, batch_size, True)
    valid_loader = make_loader(valid_set, batch_size, False)
    stored_neg_ratio = None if args.positive_only else train_set.neg_ratio
    neg_reduction_ratio = float(getattr(cfg, 'neg_reduction_ratio', 1.0))
    if not 0.0 < neg_reduction_ratio <= 1.0:
        raise ValueError('neg_reduction_ratio must be in (0, 1]')
    neg_ratio = (
        None if stored_neg_ratio is None
        else stored_neg_ratio * neg_reduction_ratio
    )
    if neg_ratio is not None:
        effective_neg = min(batch_size, max(1, int(batch_size * neg_ratio)))
        print(
            'training mix: {} positive + {} negative windows/batch | '
            'stored neg/pos {:.3f} | reduction {:.3f} | '
            'effective neg/pos {:.3f} | Zarr chunk {}'.format(
                batch_size, effective_neg, stored_neg_ratio,
                neg_reduction_ratio, neg_ratio, train_set.chunk_size()
            ),
            flush=True,
        )
    num_batch = len(train_loader)
    model = SAR()
    device = torch.device("cuda:%s" % args.gpu_idx if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    t = time.time()
    writer = SummaryWriter(log_dir=args.ckpt_dir)
    monitor = TrainingMonitor(args.ckpt_dir, 'SAR')
    best_valid_loss = float('inf')
    try:
        for epoch_idx in range(num_epochs):
          for iter_idx, (data, target) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()
            if args.positive_only:
                num_pos = data.size(0)
            else:
                data, target, num_pos = reshape_paired_batch(data, target)
            data = unfold_sar_batch(data, cfg)
            train_acc_list, train_loss = train_step(
                model, data, target, num_pos, neg_ratio, criterion, optimizer
            )
            if global_step % summary_step == 0:
                print('step {} ({}/{}) | train loss {:.2f} | {:.2f}s'.format(
                    global_step, iter_idx, epoch_idx, train_loss, time.time()-t))
                metric_text = '   frame acc. {:.2f}% | pos acc. {:.2f}%'.format(
                    100*train_acc_list[0], 100*train_acc_list[1]
                )
                if not args.positive_only:
                    metric_text += ' | neg acc. {:.2f}%'.format(100*train_acc_list[2])
                print(metric_text, flush=True)
                writer.add_scalar('loss/train_loss', train_loss, global_step)
                writer.add_scalar('frame_acc/train_acc_frame', 100*train_acc_list[0], global_step)
                writer.add_scalar('pos_acc/train_acc_pos', 100*train_acc_list[1], global_step)
                if not args.positive_only:
                    writer.add_scalar('neg_acc/train_acc_neg', 100*train_acc_list[2], global_step)
                monitor.update(global_step, {
                    'loss/train_loss': train_loss,
                    'frame_acc/train_acc_frame': 100*train_acc_list[0],
                    'pos_acc/train_acc_pos': 100*train_acc_list[1],
                    'neg_acc/train_acc_neg': (
                        None if args.positive_only else 100*train_acc_list[2]
                    ),
                })
            if global_step % valid_step_interval != 0:
                continue
            valid_acc_list, valid_loss = validate_full(
                model, valid_loader, device, cfg, criterion, args.positive_only
            )
            print('validation step {} | full-set loss {:.4f}'.format(
                global_step, valid_loss), flush=True)
            metric_text = '   frame acc. {:.2f}% | pos acc. {:.2f}%'.format(
                100*valid_acc_list[0], 100*valid_acc_list[1]
            )
            if not args.positive_only:
                metric_text += ' | neg acc. {:.2f}%'.format(100*valid_acc_list[2])
            print(metric_text, flush=True)
            writer.add_scalar('loss/valid_loss', valid_loss, global_step)
            writer.add_scalar('frame_acc/valid_acc_frame', 100*valid_acc_list[0], global_step)
            writer.add_scalar('pos_acc/valid_acc_pos', 100*valid_acc_list[1], global_step)
            if not args.positive_only:
                writer.add_scalar('neg_acc/valid_acc_neg', 100*valid_acc_list[2], global_step)
            monitor.update(global_step, {
                'loss/valid_loss': valid_loss,
                'frame_acc/valid_acc_frame': 100*valid_acc_list[0],
                'pos_acc/valid_acc_pos': 100*valid_acc_list[1],
                'neg_acc/valid_acc_neg': (
                    None if args.positive_only else 100*valid_acc_list[2]
                ),
            })
            checkpoint = os.path.join(
                args.ckpt_dir, '%s_%s-%s.ckpt' % (global_step, epoch_idx, iter_idx)
            )
            torch.save(model.state_dict(), checkpoint)
            prune_numbered_checkpoints(args.ckpt_dir, max_checkpoints)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best.ckpt'))
                print('   new best checkpoint: {} (loss {:.4f})'.format(
                    checkpoint, valid_loss), flush=True)
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


def detection_metrics(pred_class, target, num_pos):
    frame_acc = pred_class.eq(target).float().mean().item()
    detected = (pred_class == 1).any(dim=1) & (pred_class == 2).any(dim=1)
    pos_acc = detected[:num_pos].float().mean().item()
    num_neg = pred_class.size(0) - num_pos
    neg_acc = None
    if num_neg:
        neg_acc = (~detected[num_pos:]).float().mean().item()
    return [frame_acc, pos_acc, neg_acc]


def train_step(model, data, target, num_pos, neg_ratio, criterion, optimizer):
    model.train()
    available_neg = data.size(0) - num_pos
    num_neg = available_neg
    if neg_ratio is not None:
        num_neg = min(available_neg, max(1, int(num_pos * neg_ratio)))
        data = data[:num_pos + num_neg]
        target = target[:num_pos + num_neg]
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits, 2)
    loss = criterion(pred_logits.reshape(-1, 3), target.reshape(-1))
    acc_list = detection_metrics(pred_class, target, num_pos)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return acc_list, loss.item()


def valid_step(model, data, target, num_pos, criterion):
    model.eval()
    with torch.no_grad():
        pred_logits = model(data)
        pred_class = torch.argmax(pred_logits, 2)
        loss = criterion(pred_logits.reshape(-1, 3), target.reshape(-1))
        acc_list = detection_metrics(pred_class, target, num_pos)
    return acc_list, loss.item()


def validate_full(model, valid_loader, device, cfg, criterion, positive_only):
    frame_sum = 0.0
    pos_sum = 0.0
    neg_sum = 0.0
    weighted_loss = 0.0
    total_samples = 0
    total_pos = 0
    total_neg = 0
    for data, target in valid_loader:
        data = data.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()
        if positive_only:
            num_pos = data.size(0)
        else:
            data, target, num_pos = reshape_paired_batch(data, target)
        data = unfold_sar_batch(data, cfg)
        acc_list, loss = valid_step(model, data, target, num_pos, criterion)
        batch_samples = target.size(0)
        num_neg = batch_samples - num_pos
        total_samples += batch_samples
        total_pos += num_pos
        total_neg += num_neg
        weighted_loss += loss * batch_samples
        frame_sum += acc_list[0] * batch_samples
        pos_sum += acc_list[1] * num_pos
        if num_neg:
            neg_sum += acc_list[2] * num_neg
    if total_samples == 0:
        raise ValueError('validation set is empty')
    return (
        [
            frame_sum / total_samples,
            pos_sum / total_pos,
            None if total_neg == 0 else neg_sum / total_neg,
        ],
        weighted_loss / total_samples,
    )


def reshape_paired_batch(data, target):
    num_pos = data.size(0)
    data = data.transpose(0, 1)
    target = target.transpose(0, 1)
    data = data.reshape(data.size(0)*data.size(1), *data.shape[2:])
    target = target.reshape(target.size(0)*target.size(1), *target.shape[2:])
    return data, target, num_pos


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
    parser.add_argument('--positive_only', action='store_true')
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    main()
