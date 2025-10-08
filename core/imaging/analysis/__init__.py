"""
Analysis module for imaging data.

This module contains functions for extracting time series and detecting calcium events.
"""

from .timeseries import dFF, get_timestamps, z_score
from .event_detection import detect_calcium_events, binarize_calcium_event_probabilities
from .pipeline import collect_timeseries

__all__ = [
    'dFF',
    'get_timestamps', 
    'z_score',
    'detect_calcium_events',
    'binarize_calcium_event_probabilities',
    'collect_timeseries',
]
