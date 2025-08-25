# 🧠 GitHub Copilot Configuration & Guidelines for This Repository

This repository contains scientific code for neuroscience data analysis, particularly involving calcium imaging in dendritic spines, electrophysiology, and morphological reconstructions.

It is built for datasets acquired using [Scanimage](https://docs.scanimage.org/index.html). It relies on structured region metadata provided by [ROIpy](https://github.com/dcupolillo/ROIpy). It is optimized to perform semantic segmentation of dendritic spines using [DeepD3](https://github.com/ankilab/DeepD3). After trace extraction, it relies on trace analysis using [zscore_classifier](https://github.com/dcupolillo/zscore_classifier).

## Interdependence on ROIpy

Spyne is built to work in tandem with  `ROIpy` , which provides:

* ROI metadata parsing (e.g., pixel bounds, center coordinates)
* Coordinate system management (e.g., affine transformations)
* Consistent object structures for ROIs across sessions

## DeepD3 Integration

**`Spyne`** is designed to work seamlessly with segmentation outputs from  **[DeepD3](https://github.com/DeepD3/DeepD3)** —a deep learning-based tool for automated detection of dendritic spines and shafts in fluorescence microscopy images.

### Supported DeepD3 Outputs

`Spyne` expects the following `DeepD3`-generated outputs:

* **Prediction Maps** : 2D probability maps (typically in `.tif` format) for spines and shafts.
* **Spine Masks** : Label masks generated from thresholded prediction maps.
* **Metadata** : A JSON file specifying affine transforms, channel labels, and resolution used during segmentation.

### Workflow Overview

1. **Segmentation** : Use `DeepD3` to segment calcium imaging frames or averaged z-stacks.
2. **Post-processing** : performs cleanup of the binary masks.
3. **Export spines** : Convert probability maps into labeled spines.
4. **Load into ROIpy** : Parse and structure the ROIs using [ROIpy](https://github.com/yourusername/ROIpy).

## Trace Extraction & Event Classification

### Trace Extraction

`Spyne` extracts fluorescence traces from calcium imaging data by applying spine masks onto image frame sequences. The signal within each spine mask is aggregated across time, producing one trace per spine. Computes ΔF/F₀ or z-scored signals for downstream analysis.

### Event Classification with `zscore_classifier`

Spyne integrates with a modular classifier—`zscore_classifier`—to detect activity events from extracted traces. Optimized for detecting dendritic spine activation in response to localized stimulation. This tool operates on z-scored calcium signals and assigns binary labels based on thepresence of a calcium event.

#### Workflow

* Input: z-scored trace of length 50 (typical time window)
* Output: binary label (`0` or `1`) indicating presence or absence of an event
* Optional thresholding parameters can be set for different analysis conditions

#### Dependencies

* The `zscore_classifier` is implemented as a separate module for flexibility.
* Compatible with batch analysis over large datasets of traces.
* Designed for integration with Spyne’s data pipeline, but can be used independently on any z-scored trace array.

## Coding Style Preferences

**Language**: `python`

**Comments** : Concise and informative; avoid restating obvious code logic.

**New files** : Create new files only when a new logic or groupable series of functions is needed.

**Calling functions from other packages**: if a function is missing locally, but it is called from a custom python package, always assume it exists in that module

### PEP8

Always follow [PEP8](https://peps.python.org/pep-0008/) formatting rules, including but not limited to:

- `W293` blank line contains whitespace
- `W291` trailing whitespace
- `E501` line too long (< 80 characters)

### Docstring

* Always document functions, methods, and classes.
* Use **NumPy-style** docstrings.
* Include **type hints** for all parameters and return values.
* Clearly describe inputs and outputs, including units when relevant.

Example:

```python
def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.

    The return type must be duplicated in the docstring to comply
    with the NumPy docstring style.

    Parameters
    ----------
    param1
        The first parameter.
    param2
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
```

### Naming Conventions

* Use `snake_case` for variables and functions
* Use `CamelCase` for class names
* Avoid unnecessary abbreviations unless commonly accepted (e.g., `dff`, `roi`).

### Library preferences:

* Use [`flammkuchen`](https://github.com/portugueslab/flammkuchen) to handle `.h5` files.
* Use `pathlib` instead of `os` for file operations.
* Use `tifffile` for handling TIFF images.

### Optimization

Follow these best practices to write efficient and readable code:

* Use **list comprehensions** instead of `for` loops when building lists.
* Prefer **sets** or **dicts** over lists for membership checks.
* Use **built-in functions** (`sum()`, `max()`, `min()`, etc.) where possible.
* Use **generators** to handle large datasets and reduce memory usage.
* Avoid **redundant computations** inside loops; move constant expressions outside.
* Consolidate repeated logic into reusable functions.

These principles improve performance, reduce memory usage, and support clean, maintainable code.

### Data Expectations

- Traces: 1D NumPy arrays of 50 time points
- Image data: 2D or 3D arrays, dtype `np.int16` or `np.float32`

### Module Design Principles

* **Single responsibility** : Each module should do one thing
* Keep functions short (< 50 lines). If longer, break into helper functions.
* Input validation: include assertions or type checks when inputs must meet structural requirements (e.g., shape, dtype).
* Raise informative errors early (e.g., `ValueError("Trace must be length 50")`).
