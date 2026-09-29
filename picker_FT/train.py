"""Train the frame-level Transformer picker on positive/negative windows."""
import argparse
import math
import os
import time
import warnings
from contextlib import nullcontext

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from training_monitor import TrainingMonitor
from training_validation import validate_classes, validation_due
from training_loss import negative_loss_weight, describe_training_loss, weighted_window_loss, log_training_losses
from training_zarr_dataset import ValidationZarr
from training_zarr_dataset import positive_chunk_ranges
from training_zarr_dataset import ExplicitTrainingBatch, collate_training_batch, training_batch_sizes
from torch_backends import configure_torch_backends

from torch.utils.data import DataLoader, Sampler

import config
from dataset import Positive_Negative, PositiveOnly
from models import FrameTransformerPicker, count_trainable_parameters


# Also runs when spawned DataLoader workers import this module.
configure_torch_backends(config.Config())


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
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.num_samples = len(dataset)
        self.chunk_size = max(1, int(dataset.chunk_size()))
        self.chunk_ranges = positive_chunk_ranges(dataset)

    def __iter__(self):
        chunk_starts = self.chunk_ranges
        for chunk_idx in torch.randperm(len(chunk_starts)).tolist():
            start, end = chunk_starts[chunk_idx]
            batch = []
            for offset in torch.randperm(end - start).tolist():
                batch.append(start + offset)
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
        return DataLoader(
            dataset,
            batch_sampler=ChunkBatchSampler(dataset, batch_size),
            **kwargs
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def build_optimizer(model, cfg):
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
    return torch.optim.AdamW(
        parameter_groups,
        lr=float(cfg.learning_rate),
        betas=tuple(cfg.adam_betas),
        eps=float(cfg.adam_eps),
    )


def learning_rate_at_step(global_step, total_steps, cfg):
    base_lr = float(cfg.learning_rate)
    min_lr = float(cfg.min_learning_rate)
    warmup_steps = min(total_steps, max(0, int(getattr(cfg, 'warmup_steps', 10000))))
    if warmup_steps > 0 and global_step < warmup_steps:
        fraction = float(global_step) / float(max(1, warmup_steps))
        return min_lr + fraction * (base_lr - min_lr)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group['lr'] = learning_rate


def select_amp(device, enabled):
    if device.type != 'cuda' or not bool(enabled):
        return False, torch.float32
    return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device, enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def classification_loss(logits, targets):
    """Cross entropy for logits [B, 3, 246] and frame labels [B, 246]."""
    if targets.ndim != 2 or targets.size(1) != logits.size(2):
        raise ValueError(
            'Expected frame targets [B, {}], got {}'.format(
                logits.size(2), tuple(targets.shape)
            )
        )
    return F.cross_entropy(logits, targets)


def frame_metrics(logits, targets, num_pos, num_neg):
    prediction = torch.argmax(logits, dim=1)
    frame_accuracy = prediction[:num_pos].eq(targets[:num_pos]).float().mean().item() if num_pos else None
    detected = (prediction == 1).any(dim=1) & (prediction == 2).any(dim=1)
    pos_accuracy = detected[:num_pos].float().mean().item() if num_pos else None
    neg_accuracy = None
    if num_neg:
        neg_accuracy = (~detected[num_pos:num_pos + num_neg]).float().mean().item()
    return frame_accuracy, pos_accuracy, neg_accuracy


def reshape_paired_batch(data, target):
    """Flatten [batch, positive/negative, ...] with positives first."""
    num_pos = data.size(0)
    data = data.transpose(0, 1).reshape(-1, *data.shape[2:])
    target = target.transpose(0, 1).reshape(-1, *target.shape[2:])
    return data, target, num_pos


def run_step(
    model,
    data,
    target,
    optimizer,
    scaler,
    device,
    amp_enabled,
    amp_dtype,
    grad_clip_norm,
    training,
    num_pos,
    negative_weight=1.0,
    loss_details=None,
):
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    num_neg = data.size(0) - num_pos
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(data)
            loss = (weighted_window_loss(logits, target, num_pos, classification_loss,
                                         negative_weight, loss_details)
                    if training else classification_loss(logits, target))
        frame_accuracy, pos_accuracy, neg_accuracy = frame_metrics(
            logits.detach(), target, num_pos, num_neg
        )

    if training:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    return frame_accuracy, pos_accuracy, neg_accuracy, loss.item()


def main(args):
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    cfg = config.Config()
    device = torch.device('cuda:{}'.format(args.gpu_idx) if torch.cuda.is_available() else 'cpu')

    bs_pos, bs_neg = training_batch_sizes(cfg.batch_size)
    describe_training_loss(cfg, bs_pos, bs_neg)
    train_loss_details = {}
    positive_only = bs_neg == 0
    dataset_class = PositiveOnly if positive_only else Positive_Negative
    train_set = dataset_class(args.zarr_path, 'train')
    train_batches = ExplicitTrainingBatch(train_set, cfg.batch_size)
    train_loader = make_loader(train_batches, bs_pos, True, args)
    valid_loader = {kind: make_loader(
        ValidationZarr(args.zarr_path, 'frame', kind), bs_pos, False, args)
        for kind in ('positive', 'negative')}
    print('validation samples: ' + ', '.join(
        '{}={:,}'.format(kind, len(loader.dataset)) for kind, loader in valid_loader.items()), flush=True)
    print('training mix: {} positive + {} negative windows/batch | Zarr chunk {}'.format(
        bs_pos, bs_neg, train_set.chunk_size()), flush=True)
    model = FrameTransformerPicker().to(device)
    print('trainable parameters: {:,}'.format(count_trainable_parameters(model)), flush=True)
    optimizer = build_optimizer(model, cfg)
    amp_enabled, amp_dtype = select_amp(device, cfg.amp)
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    print(
        'AMP: {}{}'.format(
            'disabled' if not amp_enabled else str(amp_dtype).replace('torch.', ''),
            '' if amp_enabled else '',
        ),
        flush=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = max(1, int(cfg.num_epochs) * steps_per_epoch)
    effective_warmup_steps = min(total_steps, max(0, int(getattr(cfg, 'warmup_steps', 10000))))
    print('schedule: {:,} total steps | {:,} warmup steps'.format(
        total_steps, effective_warmup_steps
    ), flush=True)
    grad_clip_norm = float(cfg.grad_clip_norm)
    writer = SummaryWriter(log_dir=args.ckpt_dir)
    monitor = TrainingMonitor(args.ckpt_dir, 'FT')
    start_time = time.time()
    best_valid_loss = float('inf')
    max_checkpoints = int(getattr(cfg, 'max_checkpoints', 20))

    try:
        for epoch_idx in range(int(cfg.num_epochs)):
            for iter_idx, (data, target, num_pos) in enumerate(train_loader):
                global_step = steps_per_epoch * epoch_idx + iter_idx
                learning_rate = learning_rate_at_step(
                    global_step, total_steps, cfg
                )
                set_optimizer_lr(optimizer, learning_rate)
                data = data.to(device, non_blocking=True).float()
                target = target.to(device, non_blocking=True).long()
                train_frame_acc, train_pos_acc, train_neg_acc, train_loss = run_step(
                    model,
                    data,
                    target,
                    optimizer,
                    scaler,
                    device,
                    amp_enabled,
                    amp_dtype,
                    grad_clip_norm,
                    training=True,
                    num_pos=num_pos,
                    negative_weight=negative_loss_weight(cfg),
                    loss_details=train_loss_details,
                )

                if global_step % int(cfg.summary_step) == 0:
                    print(
                        'step {} ({}/{}) | lr {:.2e} | train loss {:.4f} | {:.2f}s'.format(
                            global_step, iter_idx, epoch_idx, learning_rate,
                            train_loss, time.time() - start_time,
                        ),
                        flush=True,
                    )
                    metric_text = '   frame acc. {:.2f}% | pos acc. {:.2f}%'.format(
                        100 * train_frame_acc, 100 * train_pos_acc
                    )
                    if bs_neg > 0:
                        metric_text += ' | neg acc. {:.2f}%'.format(100 * train_neg_acc)
                    print(metric_text, flush=True)
                    writer.add_scalar('loss/train_loss', train_loss, global_step)
                    loss_metrics = log_training_losses(train_loss_details, writer, global_step)
                    writer.add_scalar('frame_acc/train_frame_acc', 100 * train_frame_acc, global_step)
                    writer.add_scalar('pos_acc/train_pos_acc', 100 * train_pos_acc, global_step)
                    if bs_neg > 0:
                        writer.add_scalar('neg_acc/train_neg_acc', 100 * train_neg_acc, global_step)
                    writer.add_scalar('learning_rate', learning_rate, global_step)
                    monitor.update(global_step, {
                        'loss/train_loss': train_loss,
                        **loss_metrics,
                        'frame_acc/train_frame_acc': 100 * train_frame_acc,
                        'pos_acc/train_pos_acc': 100 * train_pos_acc,
                        'neg_acc/train_neg_acc': (
                            None if bs_neg == 0 else 100 * train_neg_acc
                        ),
                        'learning_rate': learning_rate,
                    })
                if not validation_due(global_step, total_steps, cfg.valid_step):
                    continue

                valid_frame_acc, valid_pos_acc, valid_neg_acc, valid_loss, valid_losses = validate_full(
                    model, valid_loader, optimizer, scaler, device,
                    amp_enabled, amp_dtype, grad_clip_norm, positive_only,
                )
                print(
                    'validation step {} | full-set loss {:.4f}'.format(
                        global_step, valid_loss
                    ),
                    flush=True,
                )
                metric_text = '   frame acc. {:.2f}% | pos acc. {:.2f}%'.format(
                    100 * valid_frame_acc, 100 * valid_pos_acc
                )
                if 'negative' in valid_loader:
                    metric_text += ' | neg acc. {:.2f}%'.format(100 * valid_neg_acc)
                print(metric_text, flush=True)
                for metric, value in valid_losses.items():
                    writer.add_scalar(metric, value, global_step)
                print('   ' + ' | '.join('{} {:.4f}'.format(key.split('/')[-1], value)
                                          for key, value in valid_losses.items()), flush=True)
                writer.add_scalar('loss/valid_loss', valid_loss, global_step)
                writer.add_scalar('frame_acc/valid_frame_acc', 100 * valid_frame_acc, global_step)
                writer.add_scalar('pos_acc/valid_pos_acc', 100 * valid_pos_acc, global_step)
                if 'negative' in valid_loader:
                    writer.add_scalar('neg_acc/valid_neg_acc', 100 * valid_neg_acc, global_step)
                monitor.update(global_step, {
                    'loss/valid_loss': valid_loss,
                    **valid_losses,
                    'frame_acc/valid_frame_acc': 100 * valid_frame_acc,
                    'pos_acc/valid_pos_acc': 100 * valid_pos_acc,
                    'neg_acc/valid_neg_acc': (
                        100 * valid_neg_acc
                    ),
                })
                checkpoint = os.path.join(
                    args.ckpt_dir,
                    '{}_{}-{}.ckpt'.format(global_step, epoch_idx, iter_idx),
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


def validate_full(
    model, valid_loader, optimizer, scaler, device, amp_enabled, amp_dtype,
    grad_clip_norm, positive_only,
):
    def evaluate(data, target, kind):
        data = data.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()
        num_pos = len(data) if kind == 'positive' else 0
        diagnostic, pos, neg, loss = run_step(
            model, data, target, optimizer, scaler, device, amp_enabled,
            amp_dtype, grad_clip_norm, training=False, num_pos=num_pos)
        return diagnostic, pos if kind == 'positive' else neg, loss
    return validate_classes(valid_loader, evaluate, positive_diagnostic_only=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=10)
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
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
