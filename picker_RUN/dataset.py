"""Positive/negative lazy Zarr dataset for ResUNet training."""
import os
import zarr
import numpy as np
from torch.utils.data import Dataset


def _chunk_local_negative_indices(num_pos, num_neg, pos_chunk_size, neg_chunk_size):
  if num_neg <= 0:
    raise ValueError('Positive/negative training requires at least one negative sample')
  logical_chunk = max(1, int(pos_chunk_size))
  neg_chunk_starts = np.arange(0, num_neg, max(1, int(neg_chunk_size)))
  num_blocks = int(np.ceil(num_pos / float(logical_chunk)))
  block_order = []
  while len(block_order) < num_blocks:
    block_order.extend(np.random.permutation(neg_chunk_starts).tolist())
  indices = np.empty(num_pos, dtype=np.int64)
  for block_idx in range(num_blocks):
    lo = block_idx * logical_chunk
    hi = min(num_pos, lo + logical_chunk)
    indices[lo:hi] = (
      block_order[block_idx] + np.arange(hi - lo, dtype=np.int64)
    ) % num_neg
  return indices


class Positive_Negative(Dataset):
  def __init__(self, zarr_path, zarr_group):
    self.pos_data_path = os.path.join(zarr_path, zarr_group, 'positive_data')
    self.pos_target_path = os.path.join(zarr_path, zarr_group, 'positive_target_sample')
    self.neg_data_path = os.path.join(zarr_path, zarr_group, 'negative_data')
    self.neg_target_path = os.path.join(zarr_path, zarr_group, 'negative_target_sample')
    self.pos_data = self.pos_target = None
    self.neg_data = self.neg_target = None
    pos_data = zarr.open(self.pos_data_path, mode='r')
    neg_data = zarr.open(self.neg_data_path, mode='r')
    self._length = pos_data.shape[0]
    self.neg_ratio = neg_data.shape[0] / float(self._length)
    self.neg_idx = _chunk_local_negative_indices(
        self._length,
        neg_data.shape[0],
        pos_data.chunks[0],
        neg_data.chunks[0],
    )

  def _open(self):
    if self.pos_data is None:
        self.pos_data = zarr.open(self.pos_data_path, mode='r')
        self.pos_target = zarr.open(self.pos_target_path, mode='r')
        self.neg_data = zarr.open(self.neg_data_path, mode='r')
        self.neg_target = zarr.open(self.neg_target_path, mode='r')

  def __getstate__(self):
    state = self.__dict__.copy()
    state['pos_data'] = state['pos_target'] = None
    state['neg_data'] = state['neg_target'] = None
    return state

  def __getitem__(self, index):
    self._open()
    pos_i, neg_i = index, self.neg_idx[index]
    return (
        np.array([self.pos_data[pos_i], self.neg_data[neg_i]]),
        np.array([self.pos_target[pos_i], self.neg_target[neg_i]]),
    )

  def __getitems__(self, indices):
    self._open()
    indices = np.asarray(indices)
    pos_indices = indices
    neg_indices = self.neg_idx[indices]
    pos_order = np.argsort(pos_indices)
    neg_order = np.argsort(neg_indices)
    pos_data = self.pos_data[pos_indices[pos_order].tolist()][np.argsort(pos_order)]
    pos_target = self.pos_target[pos_indices[pos_order].tolist()][np.argsort(pos_order)]
    neg_data = self.neg_data[neg_indices[neg_order].tolist()][np.argsort(neg_order)]
    neg_target = self.neg_target[neg_indices[neg_order].tolist()][np.argsort(neg_order)]
    return [
        (
            np.array([pos_data[ii], neg_data[ii]]),
            np.array([pos_target[ii], neg_target[ii]]),
        )
        for ii in range(len(indices))
    ]

  def chunk_size(self):
    self._open()
    return self.pos_data.chunks[0]

  def __len__(self):
    self._open()
    return self._length


class PositiveOnly(Dataset):
  """Positive-event subset used only by the explicit train-pos workflow."""
  def __init__(self, zarr_path, zarr_group):
    self.data_path = os.path.join(zarr_path, zarr_group, 'positive_data')
    self.target_path = os.path.join(
        zarr_path, zarr_group, 'positive_target_sample'
    )
    self.data = self.target = None
    self._length = zarr.open(self.data_path, mode='r').shape[0]

  def _open(self):
    if self.data is None:
      self.data = zarr.open(self.data_path, mode='r')
      self.target = zarr.open(self.target_path, mode='r')

  def __getstate__(self):
    state = self.__dict__.copy()
    state['data'] = state['target'] = None
    return state

  def __getitem__(self, index):
    self._open()
    return self.data[index], self.target[index]

  def __getitems__(self, indices):
    self._open()
    indices = np.asarray(indices)
    order = np.argsort(indices)
    restore = np.argsort(order)
    sorted_indices = indices[order].tolist()
    data = self.data[sorted_indices][restore]
    target = self.target[sorted_indices][restore]
    return [(data[ii], target[ii]) for ii in range(len(indices))]

  def chunk_size(self):
    self._open()
    return self.data.chunks[0]

  def __len__(self):
    return self._length
