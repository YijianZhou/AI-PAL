# References

Archived external implementations and legacy AI-PAL material retained for provenance. These sources are not part of the supported execution workflows.

## `backup/legacy_pre_reorganization/SAR_pos_legacy_remainder/run_sar/README`

Copy this directory to anywhere you like as working directory

## `backup/legacy_pre_reorganization/SAR_run_sar_generated_remainders/README`

Copy this directory to anywhere you like as working directory

## `backup/legacy_sac_preprocessing/README.md`

### Legacy SAC Preprocessing

This directory preserves the former `sac2zarr.py` and `dataset_sac.py` files
from each picker source tree. They are not used by the supported AI-PAL training
workflow, which cuts shared waveform samples into NPY shards and then builds the
shared Zarr dataset with `2_npy2zarr_eg.py`.

The original picker/preprocess layout is retained for provenance. These files
are archival snapshots; restore a pair to its original picker preprocessing
directory before attempting to run that legacy path.

## `backup/PhaseNO-1.0.1/README.md`

[![DOI](https://zenodo.org/badge/641315064.svg)](https://zenodo.org/doi/10.5281/zenodo.10224300)

### PhaseNO 
Phase Neural Operator for Multi-Station Phase Picking from Dynamic Seismic Networks.

![Method](https://github.com/sun-hongyu/PhaseNO/blob/master/phaseno.png)


#### Update Log

**Version 1.0.1 (March 31, 2025):**
- This version introduces a physical distance constraint (`dis_range`) between stations to limit information exchange to only nearby stations.
- This update significantly reduces computational cost without compromising picking performance. It better reflects realistic spatial relationships, as earthquake signals recorded at one station are most likely to appear at surrounding stations within a certain distance.
- This resolves a major limitation in PhaseNO v1.0.0, where computational cost (in terms of memory usage and speed) scaled quadratically with the number of stations. In the current version, such quadratic scaling only occurs if the user sets the distance threshold `dis_range` larger than the maximum distance between any pair of stations, effectively enabling full communication among all nodes. However, such fully connected communication is generally unnecessary.
- Users should adjust the new paremeter (`dis_range`) according to their seismic network configuration. The default is set to 30 km, meaning each station will only communicate with stations within that distance. Reducing this value can significantly lower computational demands.
- With a reasonable choice of `dis_range`, users can now include all stations in a single run, without needing to randomly select a subset from a large seismic network. 

#### Citation
```
Sun, H., Ross, Z.E., Zhu, W. and Azizzadenesheli, K., 2023. Phase Neural Operator for Multi-Station Picking of Seismic Arrivals. arXiv preprint arXiv:2305.03269.
```

#### Installation

Create an environment with conda for PhaseNO
```
conda env create -f env.yml
conda activate phaseno
```

#### Pre-trained model
Located in directory: models/*.ckpt

#### Example 
Located in directory: example

- phaseno_predict.ipynb
  
  Use the pre-trained model to pick phases from one-hour continuous data of the 2019 Ridgecrest earthquake sequence.

- phaseno_plot.ipynb
  
  Plot the predicted probabilities and picks for all stations.



## `backup/QNO-main/README.md`

### Quake Neural Operator (QNO)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20072888.svg)](https://doi.org/10.5281/zenodo.20072888)

Sun, H., 2026, Microseismic Monitoring with the Quake Neural Operator. *Nature Communications* 17, 5677. https://doi.org/10.1038/s41467-026-73965-6

#### Overview

This repository provides the official implementation of the **Quake Neural Operator (QNO)**. The QNO is a deep learning framework for automated microseismic monitoring that performs end-to-end earthquake detection and location directly from continuous seismic waveform data without explicit phase picking or phase association. Unlike conventional seismic monitoring workflows that treat detection and location as separate sequential tasks, QNO jointly learns signal classification and source characterization within a unified multi-task learning framework.

QNO enables the construction of earthquake catalogs directly from continuous waveform data and maintains robust performance under low signal-to-noise conditions where clear phase onsets may be unavailable. The framework consists of a classification task for earthquake detection and a regression task for earthquake location and origin time. For the detection task, QNO predicts the probabilities of earthquake signals and noise at each station. For the regression task, the model estimates the earthquake location and origin time directly from waveform data.

Similar to the Phase Neural Operator (Sun et al., 2023), QNO leverages neural operator architectures to model complex spatiotemporal wavefield patterns. Specifically, Fourier Neural Operators (FNOs) extract temporal features from individual station waveforms, while Graph Neural Operators (GNOs) capture spatial relationships across stations through a message-passing framework.

#### Repository Structure
This repository is organized as follows:
```
.
├── model.py               # QNO model definition
├── utils.py               # Dataset and utility functions
├── train.py               # Training script
├── evaluation.py          # Evaluation & visualization
├── data/                  # A small set of preprocessed sample data
├── output_training/       # Example training outputs
├── output_evaluation/     # Example evaluation outputs
├── requirements.txt       # Python dependencies
├── LICENSE
└── README.md
```

#### Installation
Create a Python environment named `qno` and install the required dependencies. The installation should take only a few minutes.
```
conda create -n qno python=3.9
conda activate qno
pip install -r requirements.txt
```

#### Training
Train QNO using the provided sample dataset. The provided dataset is small and training here is intended for demonstration, not for production performance.
```
python train.py \
  --sample_dir data \
  --output_name output_training \
  --epochs 10 \
  --batch_size 1
```

#### Evaluation
Run evaluation on a sample:
```
python evaluation.py \
  --checkpoint output_training/checkpoints/last.ckpt \
  --sample_file data/sample_001.h5 \
  --output_dir output_evaluation \
  --cpu
```

#### Expected output
- **Detection (per station):** predicted probabilities for earthquake signal vs. noise, where 1 indicates the highest probability of signal or noise (two channels).
- **Location (event-level):** predicted (normalized) longitude, latitude, depth, and timing.  
  Time is defined relative to the start of the input window**, assumed to span −10 s to +10 s.  
  The model outputs values in [0, 1] for regression targets; the evaluation.py converts these to physical units (longitude, latitude, depth [km], and time [s]) and plots the epicenter.
- Outputs also include input waveform visualization and spatial map of predicted vs. catalog locations

#### Data
The "data" directory contains a small number of preprocessed samples in HDF5 format. These samples are provided solely for demonstration.

Each sample includes:
* Input waveform tensor (X)
* Detection labels (Y_prob)
* Graph connectivity (edge_index)
* Location labels (Y_loc)
* Metadata (stations, event location, etc.)

#### Runtime
The runtime is approximately ~1 s per sample on an Apple M3 Max (CPU) during testing. Additional timing details are discussed in the paper (section **“Computational Cost of QNO”**).

#### Paper and Citation
If you use this code, please cite:
```
@article{sun2026qno,
  author  = {Sun, Hongyu},
  title   = {Microseismic Monitoring with the Quake Neural Operator},
  journal = {Nature Communications},
  year    = {2026},  
  volume  = {17},
  pages   = {5677},
  doi     = {10.1038/s41467-026-73965-6},
}
```

#### License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

#### Contact
Hongyu Sun
Email: hongyu-sun@outlook.com
Website: https://sun-hongyu.github.io/
