"""Training SAR model."""
import os
import shutil
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

sar_dir = '/home/zhouyj/software/2_SAR'
config_src = 'config_eg.py'
config_dst = os.path.join(sar_dir, 'config.py')
gpu_idx = 0
num_workers = 10
prefetch_factor = 2
zarr_path = '/data/bigdata/eg_train-samples.zarr'
ckpt_dir = 'output/eg_ckpt'

shutil.copyfile(config_src, config_dst)
subprocess.check_call([
    sys.executable,
    os.path.join(sar_dir, 'train.py'),
    '--gpu_idx', str(gpu_idx),
    '--num_workers', str(num_workers),
    '--prefetch_factor', str(prefetch_factor),
    '--zarr_path', zarr_path,
    '--ckpt_dir', ckpt_dir,
])