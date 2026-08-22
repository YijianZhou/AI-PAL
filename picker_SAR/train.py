"""Train SAR with both positive and negative raw-window samples."""
import os, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import torch.multiprocessing as mp
from dataset import Positive_Negative
from models import SAR
import config
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


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
    ckpt_step = cfg.ckpt_step
    batch_size = cfg.batch_size
    train_set = Positive_Negative(args.zarr_path, 'train')
    valid_set = Positive_Negative(args.zarr_path, 'valid')
    train_loader = make_loader(train_set, batch_size, True)
    valid_loader = make_loader(valid_set, batch_size, False)
    neg_ratio = train_set.neg_ratio
    effective_neg = min(batch_size, max(1, int(batch_size * neg_ratio)))
    print(
        'training mix: {} positive + {} negative windows/batch | '
        'stored neg/pos {:.3f} | Zarr chunk {}'.format(
            batch_size, effective_neg, neg_ratio, train_set.chunk_size()
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
    try:
        for epoch_idx in range(num_epochs):
          for iter_idx, (data, target) in enumerate(train_loader):
            global_step = num_batch * epoch_idx + iter_idx
            data = data.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()
            data, target = _reshape_data_target(data, target)
            data = unfold_sar_batch(data, cfg)
            train_acc_list, train_loss = train_step(model, data, target, neg_ratio, criterion, optimizer)
            if global_step % ckpt_step == 0:
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, '%s_%s-%s.ckpt' % (global_step, epoch_idx, iter_idx)))
            if global_step % summary_step != 0:
                continue
            for (data, target) in valid_loader:
                data = data.to(device, non_blocking=True).float()
                target = target.to(device, non_blocking=True).long()
                data, target = _reshape_data_target(data, target)
                data = unfold_sar_batch(data, cfg)
                valid_acc_list, valid_loss = valid_step(model, data, target, criterion)
                break
            print('step {} ({}/{}) | train loss {:.2f} | valid loss {:.2f} | {:.2f}s'.format(global_step, iter_idx, epoch_idx, train_loss, valid_loss, time.time()-t))
            acc_to_print = ''
            for ii in range(3):
                acc_to_print += ' {} acc. {:.2f}% {:.2f}% |'.format(['frame','pos','neg'][ii], 100*train_acc_list[ii], 100*valid_acc_list[ii])
            print('   %s' % acc_to_print[:-2])
            sum_loss = {'train_loss': train_loss, 'valid_loss': valid_loss}
            sum_acc_frame = {'train_acc_frame': 100*train_acc_list[0], 'valid_acc_frame': 100*valid_acc_list[0]}
            sum_acc_pos = {'train_acc_pos': 100*train_acc_list[1], 'valid_acc_pos': 100*valid_acc_list[1]}
            sum_acc_neg = {'train_acc_neg': 100*train_acc_list[2], 'valid_acc_neg': 100*valid_acc_list[2]}
            writer.add_scalars('loss', sum_loss, global_step)
            writer.add_scalars('frame_acc', sum_acc_frame, global_step)
            writer.add_scalars('pos_acc', sum_acc_pos, global_step)
            writer.add_scalars('neg_acc', sum_acc_neg, global_step)
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


def train_step(model, data, target, neg_ratio, criterion, optimizer):
    model.train()
    bs = int(target.size(0)/2)
    num_pos, num_neg = bs, int(bs*neg_ratio)
    if num_neg == 0:
        num_neg = 1
    data = data[0:num_pos+num_neg]
    target = target[0:num_pos+num_neg]
    pred_logits = model(data)
    pred_class = torch.argmax(pred_logits, 2)
    loss = criterion(pred_logits.reshape(-1, 3), target.reshape(-1))
    acc_list = []
    for ii in range(2):
        pred_i = pred_class[ii*bs : (ii+1)*bs]
        num_pos_pred = sum((pred_i == 1).any(dim=1) * (pred_i == 2).any(dim=1))
        tar_pos = target[ii*bs : (ii+1)*bs].reshape(-1)
        acc_frame = pred_i.reshape(-1).eq(tar_pos).sum() / float(tar_pos.size(0))
        if ii == 0:
            acc_list += [acc_frame, num_pos_pred/float(num_pos)]
        if ii == 1:
            acc_list += [1 - num_pos_pred/float(num_neg)]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return [acc.item() for acc in acc_list], loss.item()


def valid_step(model, data, target, criterion):
    model.eval()
    bs = int(target.size(0)/2)
    with torch.no_grad():
        pred_logits = model(data)
        pred_class = torch.argmax(pred_logits, 2)
        loss = criterion(pred_logits.reshape(-1, 3), target.reshape(-1))
        acc_list = []
        for ii in range(2):
            pred_i = pred_class[ii*bs : (ii+1)*bs]
            num_pos_pred = sum((pred_i == 1).any(dim=1) * (pred_i == 2).any(dim=1))
            tar_pos = target[ii*bs : (ii+1)*bs].reshape(-1)
            acc_frame = pred_i.reshape(-1).eq(tar_pos).sum() / float(tar_pos.size(0))
            if ii == 0:
                acc_list += [acc_frame, num_pos_pred/float(bs)]
            if ii == 1:
                acc_list += [1 - num_pos_pred/float(bs)]
    return [acc.item() for acc in acc_list], loss.item()


def _reshape_data_target(data, target):
    data = data.transpose(0, 1)
    target = target.transpose(0, 1)
    data = data.reshape(data.size(0)*data.size(1), *data.shape[2:])
    target = target.reshape(target.size(0)*target.size(1), *target.shape[2:])
    return data, target


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument(
        '--chunk_shuffle',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--zarr_path', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    main()
