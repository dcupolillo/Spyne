"""
Plotting functions for Spyne - separated from core analysis classes.

This module provides standalone plotting functions that can work with
DatasetSegmenter, RoiSegmenter, and raw data structures.

Example
-------
>>>> import spyne
>>>> import plot as plt_spyne
>>>> dataset = spyne.ImagingDataset(path/to/imaging_folder)
>>>> segmenter = spyne.DatasetSegmenter(dataset)

"""

from .plot import *
from .timeseries import *
