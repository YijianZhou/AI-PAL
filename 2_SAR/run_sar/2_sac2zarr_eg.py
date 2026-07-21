"""Compatibility wrapper. Prefer 2_npy2zarr_eg.py."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(__file__), '2_npy2zarr_eg.py'), run_name='__main__')