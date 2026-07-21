"""Dataset for SAR training with raw-waveform Zarr arrays."""
import os
import zarr
import numpy as np
from torch.utils.data import Dataset


class Positive_Negative(Dataset):
  def __init__(self, zarr_path, zarr_group):
    self.pos_data_path = os.path.join(zarr_path, zarr_group, 'positive_data')
    self.pos_tar_path = os.path.join(zarr_path, zarr_group, 'positive_target_sar')
    self.neg_data_path = os.path.join(zarr_path, zarr_group, 'negative_data')
    self.neg_tar_path = os.path.join(zarr_path, zarr_group, 'negative_target_sar')
    self.pos_data = None
    self.neg_data = None
    self.pos_tar = None
    self.neg_tar = None
    self._length = None
    self.neg_ratio = None
    self.pos_idx = None
    self.neg_idx = None
    self._init_indices()

  def _init_indices(self):
    pos_data = zarr.open(self.pos_data_path, mode='r')
    neg_data = zarr.open(self.neg_data_path, mode='r')
    num_pos, num_neg = pos_data.shape[0], neg_data.shape[0]
    self._length = num_pos
    self.neg_ratio = num_neg / num_pos
    self.pos_idx = np.random.permutation(num_pos)
    self.neg_idx = np.tile(np.arange(num_neg), int(num_pos/num_neg)+1)
    self.neg_idx = np.random.permutation(self.neg_idx)[0:num_pos]

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
    pos_i = self.pos_idx[index]
    neg_i = self.neg_idx[index]
    pos_di = self.pos_data[pos_i]
    neg_di = self.neg_data[neg_i]
    pos_ti = self.pos_tar[pos_i]
    neg_ti = self.neg_tar[neg_i]
    return np.array([pos_di, neg_di]), np.array([pos_ti, neg_ti])

  def __getitems__(self, indices):
    self._open()
    indices = np.asarray(indices)
    pos_indices = self.pos_idx[indices]
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