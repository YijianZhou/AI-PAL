"""Train PhaseNet/U-Net with positive and negative waveform windows."""
import os
import time
import argparse
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

# Fixed equal weights for noise, P, and S classes.
PHASE_CLASS_WEIGHTS = (1.0, 1.0, 1.0)


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


def main():
    torch.backends.cudnn.benchmark = True
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
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    num_batch = len(train_loader)
    t = time.time()
    writer = SummaryWriter(log_dir=args.ckpt_dir)
    monitor = TrainingMonitor(args.ckpt_dir, 'PHN')
    best_valid_loss = float('inf')
    max_checkpoints = int(getattr(cfg, 'max_checkpoints', 20))
    try:
        for epoch_idx in range(cfg.num_epochs):
          for iter_idx, (data, target, num_pos) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            train_sample_acc, train_pos_acc, train_neg_acc, train_loss = train_step(
                model, data, target, num_pos, optimizer,
                negative_loss_weight(cfg), train_loss_details
            )
            if global_step % cfg.summary_step == 0:
                print('step {} ({}/{}) | train loss {:.2f} | {:.2f}s'.format(
                    global_step, iter_idx, epoch_idx, train_loss, time.time()-t))
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
                monitor.update(global_step, {
                    'loss/train_loss': train_loss,
                    **loss_metrics,
                    'sample_acc/train_sample_acc': 100*train_sample_acc,
                    'pos_acc/train_pos_acc': 100*train_pos_acc,
                    'neg_acc/train_neg_acc': (
                        None if bs_neg == 0 else 100*train_neg_acc
                    ),
                })
            if not validation_due(global_step, cfg.num_epochs * num_batch, cfg.valid_step):
                continue
            valid_sample_acc, valid_pos_acc, valid_neg_acc, valid_loss, valid_losses = validate_full(
                model, valid_loader, device, positive_only
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
    loss_map = -torch.sum(soft_labels * log_probs, dim=1)
    class_weights = logits.new_tensor(PHASE_CLASS_WEIGHTS)
    if torch.any(class_weights != 1):
        weight_map = torch.sum(soft_labels * class_weights.view(1, -1, 1), dim=1)
        loss_map = loss_map * weight_map
    return loss_map.mean()


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


def train_step(model, data, target, num_pos, optimizer, negative_weight=1.0, loss_details=None):
    model.train()
    num_neg = data.size(0) - num_pos
    logits = model(data)
    loss = weighted_window_loss(logits, target, num_pos, soft_cross_entropy_loss,
                                negative_weight, loss_details)
    sample_acc, pos_acc, neg_acc = detection_accuracies(
        logits.detach(), target, num_pos, num_neg
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return sample_acc, pos_acc, neg_acc, loss.item()


def valid_step(model, data, target, num_pos):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = soft_cross_entropy_loss(logits, target)
        sample_acc, pos_acc, neg_acc = detection_accuracies(
            logits, target, num_pos, data.size(0) - num_pos
        )
    return sample_acc, pos_acc, neg_acc, loss.item()


def validate_full(model, valid_loader, device, positive_only):
    def evaluate(data, target, kind):
        data = data.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()
        num_pos = len(data) if kind == 'positive' else 0
        diagnostic, pos, neg, loss = valid_step(model, data, target, num_pos)
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
