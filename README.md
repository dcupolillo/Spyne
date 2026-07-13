# Spyne
[![DOI](https://sandbox.zenodo.org/badge/917193480.svg)](https://handle.test.datacite.org/10.5072/zenodo.519053)

## Description

**Spyne** is a Python package designed for advanced analysis and visualization of neuronal morphologies and dendritic spines calcium imaging data. :brain:🔬

`spyne` parses, processes, visualizes and analyzes calcium event data. It is tailored for working with comprehensive datasets from 2-photon microscopy collected using
[Vidrio ScanImage software](https://vidriotechnologies.com/).

---

## Installation

Choose the installation method based on your hardware setup:

### For GPU Users (NVIDIA CUDA)

```bash
# Clone and set up conda environment with GPU-enabled PyTorch/TensorFlow
git clone https://github.com/dcupolillo/spyne.git
cd spyne
conda env create -f environment.yml
conda activate spyne-env

# Install spyne
pip install -e .
```

**Prerequisites:** You must have an NVIDIA GPU and appropriate CUDA drivers installed. See the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for setup instructions. Ensure your CUDA version is compatible with the deep learning library versions specified in `environment.yml`.

### For CPU-Only Users

```bash
# Clone and set up conda environment with CPU-only PyTorch/TensorFlow
git clone https://github.com/dcupolillo/spyne.git
cd spyne
conda env create -f environment_CPU.yml
conda activate spyne-env-cpu

# Install spyne
pip install -e .
```

### Minimal Install (Data Processing Only)

If you only need plotting and data loading features (not spine segmentation or event classification):

```bash
pip install git+https://github.com/dcupolillo/spyne.git
```

### Feature Requirements

- **Plotting and data loading only:** Minimal install is sufficient (CPU environment).
- **Spine segmentation and calcium event classification:** GPU environment strongly recommended for reasonable performance.

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
