"""Positive-only lazy Zarr dataset for SAR training."""
import os
import zarr
import numpy as np
from torch.utils.data import Dataset


class PositiveOnly(Dataset):
  def __init__(self, zarr_path, zarr_group):
    self.data_path = os.path.join(zarr_path, zarr_group, 'positive_data')
    self.target_path = os.path.join(zarr_path, zarr_group, 'positive_target_frame')
    self.data = None
    self.target = None
    self._length = None

  def _open(self):
    if self.data is None:
        self.data = zarr.open(self.data_path, mode='r')
        self.target = zarr.open(self.target_path, mode='r')
        self._length = self.data.shape[0]

  def __getstate__(self):
    state = self.__dict__.copy()
    state['data'] = None
    state['target'] = None
    return state

  def __getitem__(self, index):
    self._open()
    return self.data[index], self.target[index]

  def __getitems__(self, indices):
    self._open()
    indices = np.asarray(indices)
    order = np.argsort(indices)
    sorted_idx = indices[order].tolist()
    data_sorted = self.data[sorted_idx]
    target_sorted = self.target[sorted_idx]
    restore = np.argsort(order)
    data = data_sorted[restore]
    target = target_sorted[restore]
    return [(data[ii], target[ii]) for ii in range(len(indices))]

  def chunk_size(self):
    self._open()
    return self.data.chunks[0]

  def __len__(self):
    self._open()
    return self._length