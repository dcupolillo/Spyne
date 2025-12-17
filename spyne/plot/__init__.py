"""
Plotting functions for Spyne - separated from core analysis classes.

This module provides standalone plotting functions that can work with
SpineDataset, RoiSpine, and raw data structures.

Example
-------
>>>> import spyne
>>>> import spyne.plot
>>>> dataset = spyne.ImagingDataset(path/to/imaging_folder)
>>>> spine_dataset = spyne.SpineDataset(dataset)
>>>>
>>>> spyne.plot.scatter(spine_dataset.spines_data)

"""

from .plot import *
