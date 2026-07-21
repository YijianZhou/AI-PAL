"""Make SAR Zarr dataset from NPY shards."""
import os
import shutil
import subprocess
import sys

out_path = '/data/bigdata/eg_train-samples.zarr'
npy_root = '/data/bigdata/eg_train-samples_npy'
sar_prep_dir = '/home/zhouyj/software/2_SAR/preprocess'
config_src = 'config_eg.py'
config_dst = os.path.join(sar_prep_dir, 'config.py')
num_workers = 10
chunk_size = 256
prefetch_factor = 1
compressor = 'lz4'
log_interval = 100000

shutil.copyfile(config_src, config_dst)
subprocess.check_call([
    sys.executable,
    os.path.join(sar_prep_dir, 'npy2zarr.py'),
    '--out_path', out_path,
    '--npy_root', npy_root,
    '--num_workers', str(num_workers),
    '--chunk_size', str(chunk_size),
    '--prefetch_factor', str(prefetch_factor),
    '--compressor', compressor,
    '--log_interval', str(log_interval),
])