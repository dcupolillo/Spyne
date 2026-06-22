"""
Denoising module for image restoration using CARE (CSBDeep).

This module provides functions for denoising calcium imaging data before
segmentation to improve detection quality.
"""

from .pipeline import denoising_pipeline, run_denoising_and_post_processing

__all__ = [
    'denoising_pipeline',
    'run_denoising_and_post_processing',
]
