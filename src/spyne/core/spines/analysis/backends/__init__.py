"""
Backend wrappers for inference engines.

This module provides unified interfaces for different neural network backends:
- DeepD3: TensorFlow-based spine segmentation
- nnU-Net: PyTorch-based semantic segmentation
- CSBDeep: CARE denoising for image restoration

Each backend runs in an isolated subprocess to avoid dependency conflicts.
"""

from .base_backend import BaseBackend, InferenceJob, InferenceResult
from .deepd3_backend import DeepD3Backend, DeepD3Job, DeepD3Result
from .nnUnet_backend import NNUNetBackend, NNUNetJob, NNUNetResult
from .csbdeep_backend import CSBDeepBackend, CSBDeepJob, CSBDeepResult

__all__ = [
    'BaseBackend',
    'InferenceJob',
    'InferenceResult',
    'DeepD3Backend',
    'DeepD3Job',
    'DeepD3Result',
    'NNUNetBackend',
    'NNUNetJob',
    'NNUNetResult',
    'CSBDeepBackend',
    'CSBDeepJob',
    'CSBDeepResult',
]
