"""Training datasets over an integrated Zarr or AWS Zarr block archive."""

import os
from pathlib import Path

import numpy as np
import zarr
from torch.utils.data import Dataset


def parse_training_blocks(blocks=None):
  if blocks is None:
    blocks = os.environ.get('AI_PAL_TRAINING_BLOCKS', '')
    if not blocks:
      blocks = os.environ.get('AI_PAL_TRAINING_YEARS', '')
  if isinstance(blocks, str):
    blocks = [value.strip() for value in blocks.split(',') if value.strip()]
  if not blocks:
    return None
  parsed = tuple(str(value).strip() for value in blocks)
  invalid = [value for value in parsed if not value or value in ('.', '..') or
             '/' in value or '\\' in value]
  if invalid:
    raise ValueError('invalid training block name(s): {}'.format(invalid))
  if len(set(parsed)) != len(parsed):
    raise ValueError('training blocks contain duplicates: {}'.format(parsed))
  return parsed


def discover_zarr_stores(zarr_path, blocks=None):
  """Return one integrated store or ordered AWS archive block stores."""
  root = Path(zarr_path)
  if (root / 'train' / 'positive_data').exists():
    if parse_training_blocks(blocks):
      raise ValueError(
        'training-block selection requires a Zarr archive, got {}'.format(root)
      )
    return [('integrated', root)]

  requested = parse_training_blocks(blocks)
  if requested is None:
    candidates = sorted(root.glob('*.zarr'))
  else:
    candidates = [root / '{}.zarr'.format(block) for block in requested]
  if not candidates:
    raise FileNotFoundError('no Zarr block stores under {}'.format(root))
  missing = [str(path) for path in candidates if not path.exists()]
  if missing:
    raise FileNotFoundError('missing Zarr block store(s): {}'.format(', '.join(missing)))
  return [(path.stem, path) for path in candidates]


def _chunk_local_negative_indices(num_pos, num_neg, chunk_size):
  if num_pos <= 0 or num_neg <= 0:
    raise ValueError('positive/negative training requires both sample classes')
  logical_chunk = max(1, int(chunk_size))
  neg_chunk_starts = np.arange(0, num_neg, logical_chunk)
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


class _TrainingZarrArrays(Dataset):
  def __init__(self, zarr_path, zarr_group, target_name, blocks=None):
    self.zarr_path = str(zarr_path)
    self.zarr_group = str(zarr_group)
    self.target_name = str(target_name)
    self.blocks = parse_training_blocks(blocks)
    self.stores = discover_zarr_stores(zarr_path, self.blocks)
    self._arrays = None

  def _metadata(self, sample_kind):
    lengths = []
    chunks = []
    trailing_shapes = []
    target_shapes = []
    for _, store in self.stores:
      data = zarr.open(
        str(store / self.zarr_group / '{}_data'.format(sample_kind)), mode='r'
      )
      target = zarr.open(
        str(store / self.zarr_group / '{}_target_{}'.format(
          sample_kind, self.target_name
        )), mode='r'
      )
      if target.shape[0] != data.shape[0]:
        raise ValueError(
          '{} data/target count mismatch in {}'.format(sample_kind, store)
        )
      lengths.append(int(data.shape[0]))
      chunks.append(int(data.chunks[0]))
      trailing_shapes.append(tuple(data.shape[1:]))
      target_shapes.append(tuple(target.shape[1:]))
    if len(set(trailing_shapes)) != 1:
      raise ValueError(
        '{} data shapes differ across archive blocks: {}'.format(
          sample_kind, trailing_shapes
        )
      )
    if len(set(target_shapes)) != 1:
      raise ValueError(
        '{} target shapes differ across archive blocks: {}'.format(
          sample_kind, target_shapes
        )
      )
    return np.asarray(lengths, dtype=np.int64), chunks

  def _open(self, sample_kind):
    if sample_kind not in ('positive', 'negative'):
      raise ValueError('unknown sample kind: {}'.format(sample_kind))
    if self._arrays is None:
      self._arrays = [{} for _ in self.stores]
    data_key = '{}_data'.format(sample_kind)
    target_key = '{}_target'.format(sample_kind)
    for store_idx, (_, store) in enumerate(self.stores):
      arrays = self._arrays[store_idx]
      if data_key in arrays:
        continue
      group = store / self.zarr_group
      arrays[data_key] = zarr.open(
        str(group / '{}_data'.format(sample_kind)), mode='r'
      )
      arrays[target_key] = zarr.open(
        str(group / '{}_target_{}'.format(sample_kind, self.target_name)),
        mode='r',
      )

  def __getstate__(self):
    state = self.__dict__.copy()
    state['_arrays'] = None
    return state

  @staticmethod
  def _locate(indices, cumulative_ends):
    indices = np.asarray(indices, dtype=np.int64)
    store_indices = np.searchsorted(cumulative_ends, indices, side='right')
    starts = np.concatenate(([0], cumulative_ends[:-1]))
    local_indices = indices - starts[store_indices]
    return store_indices, local_indices

  def _fetch_many(self, sample_kind, value_kind, indices, cumulative_ends):
    self._open(sample_kind)
    indices = np.asarray(indices, dtype=np.int64)
    store_indices, local_indices = self._locate(indices, cumulative_ends)
    result = [None] * len(indices)
    key = '{}_{}'.format(sample_kind, value_kind)
    for store_idx in np.unique(store_indices):
      positions = np.flatnonzero(store_indices == store_idx)
      local = local_indices[positions]
      order = np.argsort(local)
      values = self._arrays[int(store_idx)][key][local[order].tolist()]
      values = values[np.argsort(order)]
      for pos, value in zip(positions, values):
        result[int(pos)] = value
    return np.asarray(result)


class PositiveNegativeZarr(_TrainingZarrArrays):
  """Positive/negative pairs drawn from a virtual concatenation of blocks."""
  def __init__(self, zarr_path, zarr_group, target_name, blocks=None):
    super().__init__(zarr_path, zarr_group, target_name, blocks)
    self.pos_lengths, pos_chunks = self._metadata('positive')
    self.neg_lengths, neg_chunks = self._metadata('negative')
    self.pos_ends = np.cumsum(self.pos_lengths)
    self.neg_ends = np.cumsum(self.neg_lengths)
    self._length = int(self.pos_ends[-1])
    self.neg_ratio = int(self.neg_ends[-1]) / float(self._length)
    self._chunk_size = min(pos_chunks)
    self.neg_idx = _chunk_local_negative_indices(
      self._length, int(self.neg_ends[-1]), min(pos_chunks + neg_chunks)
    )

  def __getitem__(self, index):
    return self.__getitems__([index])[0]

  def __getitems__(self, indices):
    indices = np.asarray(indices, dtype=np.int64)
    neg_indices = self.neg_idx[indices]
    pos_data = self._fetch_many('positive', 'data', indices, self.pos_ends)
    pos_target = self._fetch_many('positive', 'target', indices, self.pos_ends)
    neg_data = self._fetch_many('negative', 'data', neg_indices, self.neg_ends)
    neg_target = self._fetch_many('negative', 'target', neg_indices, self.neg_ends)
    return [
      (
        np.array([pos_data[ii], neg_data[ii]]),
        np.array([pos_target[ii], neg_target[ii]]),
      )
      for ii in range(len(indices))
    ]

  def chunk_size(self):
    return self._chunk_size

  def __len__(self):
    return self._length


class PositiveOnlyZarr(_TrainingZarrArrays):
  def __init__(self, zarr_path, zarr_group, target_name, years=None):
    super().__init__(zarr_path, zarr_group, target_name, years)
    self.pos_lengths, chunks = self._metadata('positive')
    self.pos_ends = np.cumsum(self.pos_lengths)
    self._length = int(self.pos_ends[-1])
    self._chunk_size = min(chunks)

  def __getitem__(self, index):
    return self.__getitems__([index])[0]

  def __getitems__(self, indices):
    indices = np.asarray(indices, dtype=np.int64)
    data = self._fetch_many('positive', 'data', indices, self.pos_ends)
    target = self._fetch_many('positive', 'target', indices, self.pos_ends)
    return [(data[ii], target[ii]) for ii in range(len(indices))]

  def chunk_size(self):
    return self._chunk_size

  def __len__(self):
    return self._length
