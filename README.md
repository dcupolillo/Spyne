# Spyne

## Description

**Spyne** is a Python package designed for advanced analysis and visualization of neuronal morphologies and dendritic spines calcium imaging data. :brain::microscope:

`spyne` parses, processes, visualizes and analyzes calcium event data. It is tailored for working with comprehensive datasets from 2-photon microscopy collected using
[Vidrio ScanImage software](https://vidriotechnologies.com/).

---

## Installation

To install `spyne`, clone the repository and install the package using `pip`:

```bash
git clone https://github.com/dcupolillo/spyne.git
cd spyne
pip install -r requirements.txt
```

---

## Features

`spyne` is composed of 2 main elements:
1. **ImagingDataset** : manages the actual imaging data and metadata;
2. **DatasetSegmenter** : performs spine-wise calcium imaging analysis.

#### Dataset

The `ImagingDataset` class is the primary handler for calcium imaging data. It integrates data from 2-photon microscopy experiments, organizing data hierarchically as follows:

```mathematica
Dataset
└── ROIs
    └── Sweeps
        └── Channels
            └── Frames
```

#### Segmenter

---

## Dependencies

### ROIpy

`spyne` depends and relies on structural data generated using `ROIpy` (find repository [here](https://github.com/dcupolillo/ROIpy)).

**Example Usage:**

```python
import spyne
from pathlib import Path

date = "250101"  # YYMMDD format
cell_n = "cell0001"  #cell000n format 
data_folder = Path("data")  # where your data is stored
imaging_folder = data_folder / date / cell_n / "neuron"

# Initialize the dataset and segmenter to identify spines
dataset = spyne.ImagingDataset(imaging_folder)
segmenter = spyne.DatasetSegmenter(dataset)
```


---

## Example usage

### ImagingDataset

### DatasetSegmenter

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


