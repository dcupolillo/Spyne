"""
Analysis module for spine detection.

This module contains functions for detecting and segmenting dendritic spines.
"""

from .detection import inference
from .pipeline import semantic_segmentation_pipeline, run_inference_and_post_processing
from .post_processing import process_prediction, process_predictions
from .padding import pad_images, unpad_predictions
from .utils import (
    threshold_prediction, remove_distant_spines, remove_small_labels,
    remove_corner_joints, calculate_centroid, transform
)

__all__ = [
    'inference',
    'semantic_segmentation_pipeline',
    'run_inference_and_post_processing',
    'process_prediction',
    'process_predictions',
    'pad_images',
    'unpad_predictions',
    'threshold_prediction',
    'remove_distant_spines',
    'remove_small_labels',
    'remove_corner_joints',
    'calculate_centroid',
    'transform',
]
