"""Train modern Res-U-Net with positive and negative waveform windows."""
import os
import time
import math
import argparse
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import torch.multiprocessing as mp
from dataset import Positive_Negative, PositiveOnly
from models import UNet
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


def build_optimizer(model, cfg):
    """Build AdamW without decaying normalization parameters or biases."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith('.bias'):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    parameter_groups = [
        {'params': decay, 'weight_decay': float(cfg.weight_decay)},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return optim.AdamW(
        parameter_groups,
        lr=float(cfg.learning_rate),
        betas=tuple(cfg.adam_betas),
        eps=float(cfg.adam_eps),
    )


def learning_rate_at_step(global_step, total_steps, cfg):
    base_lr = float(cfg.learning_rate)
    min_lr = float(getattr(cfg, 'min_learning_rate', 1e-6))
    warmup_steps = min(total_steps, max(0, int(getattr(cfg, 'warmup_steps', 10000))))
    if warmup_steps > 0 and global_step < warmup_steps:
        frac = float(global_step + 1) / float(warmup_steps)
        return min_lr + frac * (base_lr - min_lr)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def select_amp(device, enabled=True):
    if device.type != 'cuda' or not bool(enabled):
        return False, torch.float32
    return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device, enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)

def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    cfg = config.Config()
    dataset_class = PositiveOnly if args.positive_only else Positive_Negative
    train_set = dataset_class(args.zarr_path, 'train')
    valid_set = dataset_class(args.zarr_path, 'valid')
    train_loader = make_loader(train_set, cfg.batch_size, True, args)
    valid_loader = make_loader(valid_set, cfg.batch_size, False, args)
    valid_iter = iter(valid_loader)
    neg_ratio = None if args.positive_only else train_set.neg_ratio
    if neg_ratio is not None:
        effective_neg = min(
            int(cfg.batch_size), max(1, int(cfg.batch_size * neg_ratio))
        )
        print(
            'training mix: {} positive + {} negative windows/batch | '
            'stored neg/pos {:.3f} | Zarr chunk {}'.format(
                cfg.batch_size, effective_neg, neg_ratio, train_set.chunk_size()
            ),
            flush=True,
        )
    model = UNet()
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    amp_enabled, amp_dtype = select_amp(device, getattr(cfg, 'amp', True))
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    print(
        'AMP: {}'.format(
            'disabled' if not amp_enabled else str(amp_dtype).replace('torch.', '')
        ),
        flush=True,
    )
    num_batch = len(train_loader)
    total_steps = max(1, cfg.num_epochs * num_batch)
    effective_warmup_steps = min(total_steps, max(0, int(getattr(cfg, 'warmup_steps', 10000))))
    print('schedule: {:,} total steps | {:,} warmup steps'.format(
        total_steps, effective_warmup_steps
    ), flush=True)
    t = time.time()
    writer = SummaryWriter(log_dir=args.ckpt_dir)
    try:
        for epoch_idx in range(cfg.num_epochs):
          for iter_idx, (data, target) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            lr = learning_rate_at_step(global_step, total_steps, cfg)
            set_optimizer_lr(optimizer, lr)
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            if args.positive_only:
                num_pos = data.size(0)
            else:
                data, target, num_pos = reshape_paired_batch(data, target)
            train_pos_acc, train_neg_acc, train_loss = train_step(
                model, data, target, num_pos, neg_ratio,
                optimizer, scaler, device, amp_enabled, amp_dtype, cfg
            )
            if global_step % cfg.ckpt_step == 0:
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'{global_step}_{epoch_idx}-{iter_idx}.ckpt'))
            if global_step % cfg.summary_step != 0:
                continue
            (data_v, target_v), valid_iter = next_valid_batch(valid_loader, valid_iter)
            data_v = data_v.to(device, non_blocking=True).float()
            target_v = target_v.to(device, non_blocking=True).float()
            if args.positive_only:
                valid_num_pos = data_v.size(0)
            else:
                data_v, target_v, valid_num_pos = reshape_paired_batch(
                    data_v, target_v
                )
            valid_pos_acc, valid_neg_acc, valid_loss = valid_step(
                model, data_v, target_v, valid_num_pos,
                device, amp_enabled, amp_dtype
            )
            print('step {} ({}/{}) | lr {:.2e} | train loss {:.4f} | valid loss {:.4f} | {:.2f}s'.format(
                global_step, iter_idx, epoch_idx, lr, train_loss, valid_loss, time.time()-t))
            metric_text = '   pos acc. {:.2f}% {:.2f}%'.format(
                100*train_pos_acc, 100*valid_pos_acc
            )
            if not args.positive_only:
                metric_text += ' | neg acc. {:.2f}% {:.2f}%'.format(
                    100*train_neg_acc, 100*valid_neg_acc
                )
            print(metric_text)
            writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, global_step)
            writer.add_scalars('pos_acc', {'train_pos_acc': 100*train_pos_acc, 'valid_pos_acc': 100*valid_pos_acc}, global_step)
            if not args.positive_only:
                writer.add_scalars('neg_acc', {'train_neg_acc': 100*train_neg_acc, 'valid_neg_acc': 100*valid_neg_acc}, global_step)
            writer.add_scalar('learning_rate', lr, global_step)
    finally:
        writer.close()

def soft_cross_entropy_loss(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.sum(soft_labels * log_probs, dim=1).mean()


def detection_accuracies(logits, num_pos, num_neg):
    pred = torch.argmax(logits, 1)
    detected = (pred == 1).any(dim=1) & (pred == 2).any(dim=1)
    pos_accuracy = detected[:num_pos].float().mean()
    neg_accuracy = None
    if num_neg:
        neg_accuracy = (~detected[num_pos:num_pos + num_neg]).float().mean().item()
    return pos_accuracy.item(), neg_accuracy


def reshape_paired_batch(data, target):
    """Flatten [batch, positive/negative, ...] with positives first."""
    num_pos = data.size(0)
    data = data.transpose(0, 1).reshape(-1, *data.shape[2:])
    target = target.transpose(0, 1).reshape(-1, *target.shape[2:])
    return data, target, num_pos


def train_step(
    model, data, target, num_pos, neg_ratio,
    optimizer, scaler, device, amp_enabled, amp_dtype, cfg,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    available_neg = data.size(0) - num_pos
    num_neg = available_neg
    if neg_ratio is not None:
        num_neg = min(available_neg, max(1, int(num_pos * neg_ratio)))
        data = torch.cat((data[:num_pos], data[num_pos:num_pos + num_neg]))
        target = torch.cat((target[:num_pos], target[num_pos:num_pos + num_neg]))
    with autocast_context(device, amp_enabled, amp_dtype):
        logits = model(data)
        loss = soft_cross_entropy_loss(logits.float(), target)
    pos_acc, neg_acc = detection_accuracies(logits.detach(), num_pos, num_neg)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_clip_norm = float(getattr(cfg, 'grad_clip_norm', 0.0))
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    return pos_acc, neg_acc, loss.item()


def valid_step(model, data, target, num_pos, device, amp_enabled, amp_dtype):
    model.eval()
    with torch.inference_mode():
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(data)
            loss = soft_cross_entropy_loss(logits.float(), target)
        pos_acc, neg_acc = detection_accuracies(
            logits, num_pos, data.size(0) - num_pos
        )
    return pos_acc, neg_acc, loss.item()


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
