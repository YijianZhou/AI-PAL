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
import warnings
warnings.filterwarnings("ignore")

# Fixed equal weights for noise, P, and S classes.
PHASE_CLASS_WEIGHTS = (1.0, 1.0, 1.0)

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
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    num_batch = len(train_loader)
    t = time.time()
    writer = SummaryWriter(log_dir=args.ckpt_dir)
    try:
        for epoch_idx in range(cfg.num_epochs):
          for iter_idx, (data, target) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            if args.positive_only:
                num_pos = data.size(0)
            else:
                data, target, num_pos = reshape_paired_batch(data, target)
            train_pos_acc, train_neg_acc, train_loss = train_step(
                model, data, target, num_pos, neg_ratio, optimizer
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
                model, data_v, target_v, valid_num_pos
            )
            print('step {} ({}/{}) | train loss {:.2f} | valid loss {:.2f} | {:.2f}s'.format(
                global_step, iter_idx, epoch_idx, train_loss, valid_loss, time.time()-t))
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


def train_step(model, data, target, num_pos, neg_ratio, optimizer):
    model.train()
    available_neg = data.size(0) - num_pos
    num_neg = available_neg
    if neg_ratio is not None:
        num_neg = min(available_neg, max(1, int(num_pos * neg_ratio)))
        data = torch.cat((data[:num_pos], data[num_pos:num_pos + num_neg]))
        target = torch.cat((target[:num_pos], target[num_pos:num_pos + num_neg]))
    logits = model(data)
    loss = soft_cross_entropy_loss(logits, target)
    pos_acc, neg_acc = detection_accuracies(logits.detach(), num_pos, num_neg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return pos_acc, neg_acc, loss.item()


def valid_step(model, data, target, num_pos):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = soft_cross_entropy_loss(logits, target)
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
