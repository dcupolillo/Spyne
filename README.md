# Spyne

## Description

**Spyne** is a Python package designed for advanced analysis and visualization of neuronal morphologies and dendritic spines calcium imaging data. :brain:🔬

`spyne` parses, processes, visualizes and analyzes calcium event data. It is tailored for working with comprehensive datasets from 2-photon microscopy collected using
[Vidrio ScanImage software](https://vidriotechnologies.com/).

---

## Installation

### Note to GPU users

To use the GPU environment, you must have an NVIDIA GPU and the appropriate CUDA drivers installed. See the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for instructions. Make sure your CUDA version is compatible with the versions of deep learning libraries specified in `environment.yml`.

### How to install:

To install `spyne`, clone the repository and create a conda environment using the appropriate YAML file for your hardware:

```bash
git clone https://github.com/dcupolillo/spyne.git
cd spyne

# For CPU-only users:
conda env create -f environment_CPU.yml

# For GPU users (NVIDIA CUDA installed):
conda env create -f environment.yml

# Activate env
conda activate spyne-env  # or spyne-env-cpu for CPU-only
```

**CPU vs GPU environments:**

- `environment_CPU.yml` is for users without a dedicated GPU. It installs CPU-compatible versions of all dependencies.
- `environment.yml` is for users with an NVIDIA GPU and CUDA drivers. It installs GPU-accelerated packages (e.g., tensorflow-gpu).

If you only want to use plotting and data loading features (not deep learning/segmentation), the CPU environment is sufficient. To perform analysis such as spine segmentation and calcium event classification, GPU is required.

---

## Features

`spyne` is composed of 3 main elements:

1. **ImagingDataset** : manages the actual imaging data and metadata;
2. **SpineDataset** : performs spine segmentation and calcium imaging analysis.
3. **EphyDataset**: manages and analyzes patch-clamp data.

#### Imaging dataset

The `ImagingDataset` class is the primary handler for calcium imaging data. It integrates data from 2-photon microscopy experiments, organizing data hierarchically as follows:

```mathematica
Dataset
└── ROIs
    └── Sweeps
        └── Channels
            └── Frames
```

#### Spine dataset

spine dataset

#### Ephy dataset

ephy dataset

---

## Dependencies

### ROIpy

`spyne` depends and relies on structural data generated using `ROIpy` (find repository [here](https://github.com/dcupolillo/ROIpy)).

**Example Usage:**

```python
import spyne
from pathlib import Path

data_folder = Path("data")  # where your data is stored
date = "250101"  # YYMMDD format
cell_n = "cell0001"  #cell000n format 
imaging_folder = data_folder / date / cell_n / "neuron"

# Initialize the dataseta
dataset = spyne.ImagingDataset(imaging_folder)
spine_dataset = spyne.SpineDataset(dataset)
ephy_dataset = spyne.EphyDataset(dataset)
```

---

## Example usage

### ImagingDataset

### SpineDataset

## Key Features

- **Calcium Imaging Data Processing**:

  - Handle dF/F traces and z-scores.
  - Detect and binarize calcium events with customizable thresholds.
  - Segment calcium traces into spines and neurites.
- **Region of Interest (ROI) Analysis**:

  - Automatically generate and optimize rectangular ROIs for scanning.
  - Assign ROIs to specific neuronal branches and compute branch attributes.
  - Handle transformation matrices for pixel-to-reference coordinate mapping.
- **Visualization**:

  - Plot neuronal structures with color-coded branches.
  - Visualize calcium traces for individual spines and neurites.
  - Generate ROI layouts for scan fields.

---
