"""
Preprocessing module for imaging data.

This module contains functions for image filtering, denoising, and artifact correction.
"""

# Import main filtering functions
from .filters import median, gaussian
# Import denoising functions
from .denoise import (
    radius_to_kernel_size, 
    compute_local_contrast, 
    compute_local_noise,
    adaptive_3d_median_filter
)

# Import artifact correction functions
from .artifacts import (
    modify_frames, 
    identify_rows_deviation,
    collect_deviating_rows_for_channels
)

__all__ = [
    # Filters
    'median',
    'gaussian', 
    'modified_okada_filter',
    'ewma',
    # Denoising
    'radius_to_kernel_size',
    'compute_local_contrast',
    'compute_local_noise', 
    'adaptive_3d_median_filter',
    # Artifacts
    'modify_frames',
    'identify_rows_deviation',
    'collect_deviating_rows_for_channels',
]