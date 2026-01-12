"""
Analysis module for spine detection.

This module contains functions for detecting and segmenting dendritic spines.
"""

from deepd3.inference.detection import inference
from deepd3.inference.post_processing import process_prediction, process_predictions
from deepd3.inference.post_processing_utils import (
    threshold_prediction, remove_distant_spines, remove_small_labels,
    remove_corner_joints, calculate_centroid, transform
)
from .segmentation.pipeline import semantic_segmentation_pipeline, run_inference_and_post_processing
from deepd3.inference.utils import pad_images, unpad_predictions
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
