"""Integrated-local and AWS-block Zarr datasets for Frame Transformer training."""
from training_zarr_dataset import PositiveNegativeZarr, PositiveOnlyZarr


class Positive_Negative(PositiveNegativeZarr):
  def __init__(self, zarr_path, zarr_group):
    super().__init__(zarr_path, zarr_group, 'frame')


class PositiveOnly(PositiveOnlyZarr):
  def __init__(self, zarr_path, zarr_group):
    super().__init__(zarr_path, zarr_group, 'frame')
