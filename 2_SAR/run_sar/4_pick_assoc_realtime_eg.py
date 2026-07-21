"""Pseudo-realtime multi-model picking + PAL association.

Run this file from the run_sar workdir after preparing:
  - input/station_realtime_scsn_complete_r*.csv
  - a trained SAR checkpoint directory
  - incoming miniSEED files in /app/aqms/ai_pal/IN
"""
import os
import shutil


# i/o paths
sar_dir = "/home/zhouyj/software/2_SAR"
shutil.copyfile("config_realtime_eg.py", os.path.join(sar_dir, "config.py"))

subnet_sta_files = [
    "input/station_realtime_scsn_complete_r1.csv",
    "input/station_realtime_scsn_complete_r2.csv",
    "input/station_realtime_scsn_complete_r3.csv",
    "input/station_realtime_scsn_complete_r4.csv",
    "input/station_realtime_scsn_complete_r5.csv",
]
in_dir = "/app/aqms/ai_pal/IN"
in_glob = "*.ms"
out_root = "output/realtime"

# Model and GPU controls
pg1_gpu_idx = 0
pg2_gpu_idx = 1
num_workers = 5
ckpt_dir = "input/EG_ckpt"
ckpt_idx = -1

os.system(
    "python -u {}/run_realtime.py --subnet_sta_files {} --in_dir={} "
    "--in_glob={} --out_root={} --pg1_gpu_idx={} --pg2_gpu_idx={} "
    "--num_workers={} --ckpt_dir={} --ckpt_idx={}".format(
        sar_dir,
        " ".join(subnet_sta_files),
        in_dir,
        in_glob,
        out_root,
        pg1_gpu_idx,
        pg2_gpu_idx,
        num_workers,
        ckpt_dir,
        ckpt_idx,
    )
)
