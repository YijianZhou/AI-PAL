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
from training_monitor import TrainingMonitor
from training_validation import validate_classes, validation_due
from training_loss import negative_loss_weight, describe_training_loss, weighted_window_loss, log_training_losses
from training_zarr_dataset import ValidationZarr
from training_zarr_dataset import positive_chunk_ranges
from training_zarr_dataset import ExplicitTrainingBatch, collate_training_batch, training_batch_sizes
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
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        self.chunk_size = max(1, int(dataset.chunk_size()))
        self.chunk_ranges = positive_chunk_ranges(dataset)

    def __iter__(self):
        chunk_starts = self.chunk_ranges
        perm = torch.randperm(len(chunk_starts)).tolist()
        for chunk_idx in perm:
            start, end = chunk_starts[chunk_idx]
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
        for start, end in self.chunk_ranges:
            chunk_samples = end - start
            if self.drop_last:
                total_batches += chunk_samples // self.batch_size
            else:
                total_batches += (
                    chunk_samples + self.batch_size - 1
                ) // self.batch_size
        return total_batches


def make_loader(dataset, batch_size, shuffle, args):
    kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if isinstance(dataset, ExplicitTrainingBatch):
        kwargs['collate_fn'] = collate_training_batch
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    if shuffle and args.chunk_shuffle:
        return DataLoader(dataset, batch_sampler=ChunkBatchSampler(dataset, batch_size), **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


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
    bs_pos, bs_neg = training_batch_sizes(cfg.batch_size)
    describe_training_loss(cfg, bs_pos, bs_neg)
    train_loss_details = {}
    positive_only = bs_neg == 0
    dataset_class = PositiveOnly if positive_only else Positive_Negative
    train_set = dataset_class(args.zarr_path, 'train')
    train_batches = ExplicitTrainingBatch(train_set, cfg.batch_size)
    train_loader = make_loader(train_batches, bs_pos, True, args)
    valid_loader = {kind: make_loader(
        ValidationZarr(args.zarr_path, 'sample', kind), bs_pos, False, args)
        for kind in ('positive', 'negative')}
    print('validation samples: ' + ', '.join(
        '{}={:,}'.format(kind, len(loader.dataset)) for kind, loader in valid_loader.items()), flush=True)
    print('training mix: {} positive + {} negative windows/batch | Zarr chunk {}'.format(
        bs_pos, bs_neg, train_set.chunk_size()), flush=True)
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
    monitor = TrainingMonitor(args.ckpt_dir, 'RUN')
    best_valid_loss = float('inf')
    max_checkpoints = int(getattr(cfg, 'max_checkpoints', 20))
    try:
        for epoch_idx in range(cfg.num_epochs):
          for iter_idx, (data, target, num_pos) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            lr = learning_rate_at_step(global_step, total_steps, cfg)
            set_optimizer_lr(optimizer, lr)
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            train_sample_acc, train_pos_acc, train_neg_acc, train_loss = train_step(
                model, data, target, num_pos,
                optimizer, scaler, device, amp_enabled, amp_dtype, cfg, train_loss_details
            )
            if global_step % cfg.summary_step == 0:
                print('step {} ({}/{}) | lr {:.2e} | train loss {:.4f} | {:.2f}s'.format(
                    global_step, iter_idx, epoch_idx, lr, train_loss, time.time()-t))
                metric_text = '   sample acc. {:.2f}% | pos acc. {:.2f}%'.format(
                    100*train_sample_acc, 100*train_pos_acc
                )
                if bs_neg > 0:
                    metric_text += ' | neg acc. {:.2f}%'.format(100*train_neg_acc)
                print(metric_text, flush=True)
                writer.add_scalar('loss/train_loss', train_loss, global_step)
                loss_metrics = log_training_losses(train_loss_details, writer, global_step)
                writer.add_scalar(
                    'sample_acc/train_sample_acc', 100*train_sample_acc, global_step
                )
                writer.add_scalar('pos_acc/train_pos_acc', 100*train_pos_acc, global_step)
                if bs_neg > 0:
                    writer.add_scalar('neg_acc/train_neg_acc', 100*train_neg_acc, global_step)
                writer.add_scalar('learning_rate', lr, global_step)
                monitor.update(global_step, {
                    'loss/train_loss': train_loss,
                    **loss_metrics,
                    'sample_acc/train_sample_acc': 100*train_sample_acc,
                    'pos_acc/train_pos_acc': 100*train_pos_acc,
                    'neg_acc/train_neg_acc': (
                        None if bs_neg == 0 else 100*train_neg_acc
                    ),
                    'learning_rate': lr,
                })
            if not validation_due(global_step, cfg.num_epochs * num_batch, cfg.valid_step):
                continue
            valid_sample_acc, valid_pos_acc, valid_neg_acc, valid_loss, valid_losses = validate_full(
                model, valid_loader, device, amp_enabled, amp_dtype,
                positive_only
            )
            print('validation step {} | full-set loss {:.4f}'.format(
                global_step, valid_loss), flush=True)
            metric_text = '   sample acc. {:.2f}% | pos acc. {:.2f}%'.format(
                100*valid_sample_acc, 100*valid_pos_acc
            )
            if 'negative' in valid_loader:
                metric_text += ' | neg acc. {:.2f}%'.format(100*valid_neg_acc)
            print(metric_text, flush=True)
            for metric, value in valid_losses.items():
                writer.add_scalar(metric, value, global_step)
            print('   ' + ' | '.join('{} {:.4f}'.format(key.split('/')[-1], value)
                                      for key, value in valid_losses.items()), flush=True)
            writer.add_scalar('loss/valid_loss', valid_loss, global_step)
            writer.add_scalar(
                'sample_acc/valid_sample_acc', 100*valid_sample_acc, global_step
            )
            writer.add_scalar('pos_acc/valid_pos_acc', 100*valid_pos_acc, global_step)
            if 'negative' in valid_loader:
                writer.add_scalar('neg_acc/valid_neg_acc', 100*valid_neg_acc, global_step)
            monitor.update(global_step, {
                'loss/valid_loss': valid_loss,
                **valid_losses,
                'sample_acc/valid_sample_acc': 100*valid_sample_acc,
                'pos_acc/valid_pos_acc': 100*valid_pos_acc,
                'neg_acc/valid_neg_acc': (
                    100*valid_neg_acc
                ),
            })
            checkpoint = os.path.join(
                args.ckpt_dir, f'{global_step}_{epoch_idx}-{iter_idx}.ckpt'
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

def soft_cross_entropy_loss(logits, soft_labels):
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.sum(soft_labels * log_probs, dim=1).mean()


def detection_accuracies(logits, target, num_pos, num_neg):
    pred = torch.argmax(logits, 1)
    target_class = torch.argmax(target, 1)
    sample_accuracy = pred.eq(target_class).float().mean()
    detected = (pred == 1).any(dim=1) & (pred == 2).any(dim=1)
    pos_accuracy = detected[:num_pos].float().mean().item() if num_pos else None
    neg_accuracy = None
    if num_neg:
        neg_accuracy = (~detected[num_pos:num_pos + num_neg]).float().mean().item()
    return sample_accuracy.item(), pos_accuracy, neg_accuracy


def reshape_paired_batch(data, target):
    """Flatten [batch, positive/negative, ...] with positives first."""
    num_pos = data.size(0)
    data = data.transpose(0, 1).reshape(-1, *data.shape[2:])
    target = target.transpose(0, 1).reshape(-1, *target.shape[2:])
    return data, target, num_pos


def train_step(
    model, data, target, num_pos,
    optimizer, scaler, device, amp_enabled, amp_dtype, cfg,
    loss_details=None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_neg = data.size(0) - num_pos
    with autocast_context(device, amp_enabled, amp_dtype):
        logits = model(data)
        loss = weighted_window_loss(logits.float(), target, num_pos, soft_cross_entropy_loss,
                                    negative_loss_weight(cfg), loss_details)
    sample_acc, pos_acc, neg_acc = detection_accuracies(
        logits.detach(), target, num_pos, num_neg
    )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_clip_norm = float(getattr(cfg, 'grad_clip_norm', 0.0))
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    return sample_acc, pos_acc, neg_acc, loss.item()


def valid_step(model, data, target, num_pos, device, amp_enabled, amp_dtype):
    model.eval()
    with torch.inference_mode():
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(data)
            loss = soft_cross_entropy_loss(logits.float(), target)
        sample_acc, pos_acc, neg_acc = detection_accuracies(
            logits, target, num_pos, data.size(0) - num_pos
        )
    return sample_acc, pos_acc, neg_acc, loss.item()


def validate_full(
    model, valid_loader, device, amp_enabled, amp_dtype, positive_only,
):
    def evaluate(data, target, kind):
        data = data.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()
        num_pos = len(data) if kind == 'positive' else 0
        diagnostic, pos, neg, loss = valid_step(model, data, target, num_pos, device, amp_enabled, amp_dtype)
        return diagnostic, pos if kind == 'positive' else neg, loss
    return validate_classes(valid_loader, evaluate)


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
