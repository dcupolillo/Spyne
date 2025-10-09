"""
Analysis module for spine detection.

This module contains functions for detecting and segmenting dendritic spines.
"""

from .segmentation.detection import inference
from .segmentation.pipeline import semantic_segmentation_pipeline, run_inference_and_post_processing
from .segmentation.post_processing import process_prediction, process_predictions
from .segmentation.padding import pad_images, unpad_predictions
from .segmentation.utils import (
    threshold_prediction, remove_distant_spines, remove_small_labels,
    remove_corner_joints, calculate_centroid, transform
)
from .timeseries.pipeline import collect_timeseries
from .timeseries.event_detection import detect_calcium_events, binarize_calcium_event_probabilities
from .timeseries.timeseries import dFF, get_timestamps, z_score
from .timeseries.filters import modified_okada_filter, ewma

__all__ = [
    # Segmentation
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
    # Timeseries
    'collect_timeseries',
    'detect_calcium_events',
    'binarize_calcium_event_probabilities',
    'dFF',
    'get_timestamps',
    'z_score',
    'modified_okada_filter',
    'ewma',
]
