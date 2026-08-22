"""Dataset for SAR training with raw-waveform Zarr arrays."""
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
    self.pos_tar_path = os.path.join(zarr_path, zarr_group, 'positive_target_frame')
    self.neg_data_path = os.path.join(zarr_path, zarr_group, 'negative_data')
    self.neg_tar_path = os.path.join(zarr_path, zarr_group, 'negative_target_frame')
    self.pos_data = None
    self.neg_data = None
    self.pos_tar = None
    self.neg_tar = None
    self._length = None
    self.neg_ratio = None
    self.neg_idx = None
    self._init_indices()

  def _init_indices(self):
    pos_data = zarr.open(self.pos_data_path, mode='r')
    neg_data = zarr.open(self.neg_data_path, mode='r')
    num_pos, num_neg = pos_data.shape[0], neg_data.shape[0]
    self._length = num_pos
    self.neg_ratio = num_neg / num_pos
    self.neg_idx = _chunk_local_negative_indices(
      num_pos, num_neg, pos_data.chunks[0], neg_data.chunks[0]
    )

  def _open(self):
    if self.pos_data is None:
        self.pos_data = zarr.open(self.pos_data_path, mode='r')
        self.neg_data = zarr.open(self.neg_data_path, mode='r')
        self.pos_tar = zarr.open(self.pos_tar_path, mode='r')
        self.neg_tar = zarr.open(self.neg_tar_path, mode='r')

  def __getstate__(self):
    state = self.__dict__.copy()
    state['pos_data'] = None
    state['neg_data'] = None
    state['pos_tar'] = None
    state['neg_tar'] = None
    return state

  def __getitem__(self, index):
    self._open()
    pos_i = index
    neg_i = self.neg_idx[index]
    pos_di = self.pos_data[pos_i]
    neg_di = self.neg_data[neg_i]
    pos_ti = self.pos_tar[pos_i]
    neg_ti = self.neg_tar[neg_i]
    return np.array([pos_di, neg_di]), np.array([pos_ti, neg_ti])

  def __getitems__(self, indices):
    self._open()
    indices = np.asarray(indices)
    pos_indices = indices
    neg_indices = self.neg_idx[indices]
    pos_order = np.argsort(pos_indices)
    neg_order = np.argsort(neg_indices)
    pos_sorted = pos_indices[pos_order].tolist()
    neg_sorted = neg_indices[neg_order].tolist()
    pos_data_sorted = self.pos_data[pos_sorted]
    pos_tar_sorted = self.pos_tar[pos_sorted]
    neg_data_sorted = self.neg_data[neg_sorted]
    neg_tar_sorted = self.neg_tar[neg_sorted]
    pos_restore = np.argsort(pos_order)
    neg_restore = np.argsort(neg_order)
    pos_data = pos_data_sorted[pos_restore]
    pos_tar = pos_tar_sorted[pos_restore]
    neg_data = neg_data_sorted[neg_restore]
    neg_tar = neg_tar_sorted[neg_restore]
    return [
        (np.array([pos_data[ii], neg_data[ii]]), np.array([pos_tar[ii], neg_tar[ii]]))
        for ii in range(len(indices))
    ]

  def chunk_size(self):
    self._open()
    return self.pos_data.chunks[0]

  def __len__(self):
    return self._length
