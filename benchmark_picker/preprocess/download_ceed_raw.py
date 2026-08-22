"""Download CEED raw files from Hugging Face.

The AI4EPS/CEED dataset script points to waveform HDF5 files stored in the
AI4EPS/quakeflow_nc and AI4EPS/quakeflow_sc dataset repositories. This script
keeps the raw files on disk so later preprocessing can read HDF5 directly.
"""

import os
from importlib.util import find_spec
from pathlib import Path

from huggingface_hub import snapshot_download


# i/o paths
root_dir = Path("/nas/zhouyj/CEED")
ceed_meta_dir = root_dir / "AI4EPS_CEED"
nc_waveform_dir = root_dir / "quakeflow_nc"
sc_waveform_dir = root_dir / "quakeflow_sc"

# repo names
ceed_repo = "AI4EPS/CEED"
nc_repo = "AI4EPS/quakeflow_nc"
sc_repo = "AI4EPS/quakeflow_sc"

# download controls
resume_download = True
max_workers = 8

# If you only want a quick smoke test, set these to e.g. ["waveform_h5/2023.h5"]
# and ["waveform_h5/2023.h5"]. Keep None for the full dataset.
nc_allow_patterns = ["waveform_h5/*.h5"]
sc_allow_patterns = ["waveform_h5/*.h5"]
ceed_allow_patterns = [
    "*.py",
    "*.md",
    "*.csv",
    "*.ipynb",
]

# Optional faster downloader supported by huggingface_hub when hf_transfer is installed.
# You can also set this in the shell before running the script.
if find_spec("hf_transfer") is not None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_repo(repo_id, local_dir, allow_patterns):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir}")
    out = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=resume_download,
        max_workers=max_workers,
    )
    print(f"Finished {repo_id}: {out}")


def main():
    root_dir.mkdir(parents=True, exist_ok=True)

    download_repo(ceed_repo, ceed_meta_dir, ceed_allow_patterns)
    download_repo(nc_repo, nc_waveform_dir, nc_allow_patterns)
    download_repo(sc_repo, sc_waveform_dir, sc_allow_patterns)

    print("Done.")
    print(f"Metadata: {ceed_meta_dir}")
    print(f"NC HDF5:   {nc_waveform_dir / 'waveform_h5'}")
    print(f"SC HDF5:   {sc_waveform_dir / 'waveform_h5'}")


if __name__ == "__main__":
    main()
