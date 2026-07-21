"""Cut SAR training samples directly into NPY shards."""
import os
import shutil
import subprocess
import sys

# i/o paths
data_dir = '/data/Example_data'
sar_prep_dir = '/home/zhouyj/software/2_SAR/preprocess'
config_src = 'config_eg.py'
config_dst = os.path.join(sar_prep_dir, 'config.py')
fpha = 'input/eg_pal_hyp.pha'
fpick = 'input/eg_pal.pick'  # all PAL picks
out_root = '/data/bigdata/eg_train-samples_npy'
num_workers = 10
shard_size = 1024

shutil.copyfile(config_src, config_dst)
subprocess.check_call([
    sys.executable,
    os.path.join(sar_prep_dir, 'cut_positive_npy.py'),
    '--data_dir', data_dir,
    '--fpha', fpha,
    '--out_root', out_root,
    '--num_workers', str(num_workers),
    '--shard_size', str(shard_size),
])
subprocess.check_call([
    sys.executable,
    os.path.join(sar_prep_dir, 'cut_negative_npy.py'),
    '--data_dir', data_dir,
    '--fpha', fpha,
    '--fpick', fpick,
    '--out_root', out_root,
    '--num_workers', str(num_workers),
    '--shard_size', str(shard_size),
])