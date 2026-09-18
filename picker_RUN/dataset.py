"""Integrated-local and AWS-block Zarr datasets for ResUNet training."""
from training_zarr_dataset import PositiveNegativeZarr, PositiveOnlyZarr


class Positive_Negative(PositiveNegativeZarr):
  def __init__(self, zarr_path, zarr_group):
    super().__init__(zarr_path, zarr_group, 'sample')


class PositiveOnly(PositiveOnlyZarr):
  def __init__(self, zarr_path, zarr_group):
    super().__init__(zarr_path, zarr_group, 'sample')
