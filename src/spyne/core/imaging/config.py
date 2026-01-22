"""Configuration management for imaging datasets.

Created on January 21, 2026
@author: dcupolillo
"""

from __future__ import annotations
from pathlib import Path
import yaml
import tensorflow as tf


class ImagingDatasetConfig:
    """
    Configuration manager for imaging dataset processing.
    
    Handles all configuration parameters for imaging data loading,
    processing, and validation. Supports YAML-based configuration
    with parameter override capabilities.
    """
    
    def __init__(
        self,
        config_path: str or Path = None,
        kernel_size_um: tuple = None,
        pmt_artifact_detection_threshold: int = None,
        **overrides
    ):
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_path : str | Path, optional
            Path to configuration YAML file. If None, uses default config.
        kernel_size_um : tuple, optional
            3D median filter kernel size. Overrides config file value.
        pmt_artifact_detection_threshold : int, optional
            PMT artifact detection threshold. Overrides config file value.
        **overrides
            Additional parameter overrides.
        """
        # Load base configuration
        self._config = self._load_config(config_path)
        
        # Apply parameter overrides
        if kernel_size_um is not None:
            self._config['processing']['kernel_size_um'] = kernel_size_um
        
        if pmt_artifact_detection_threshold is not None:
            self._config['processing']['pmt_artifact_detection_threshold'] = pmt_artifact_detection_threshold
        
        # Apply any additional overrides
        for key, value in overrides.items():
            if key in self._config['processing']:
                self._config['processing'][key] = value
        
        # Validate configuration
        self._validate_config()
        
        # Detect hardware
        self._device = self._set_device()
    
    def _load_config(
            self,
            config_path: str or Path = None
    ) -> dict:
        """Load configuration from YAML file."""
        
        if config_path is None:
            # Use default config in package
            config_path = Path(__file__).parent.parent /"config" / "imaging_config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        
        # Validate kernel size
        kernel = self._config['processing']['kernel_size_um']
        if not isinstance(kernel, (list, tuple)) or len(kernel) != 3:
            raise ValueError("kernel_size_um must be a tuple/list of length 3")
        
        # Validate individual kernel dimensions
        x, y, t = kernel
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("Spatial kernel sizes must be numeric")
        if not isinstance(t, int) or t <= 0:
            raise ValueError("Temporal kernel size must be a positive integer")
        if int(x) % 2 == 0 or int(y) % 2 == 0:
            raise ValueError("Spatial kernel sizes must be odd integers")
        
        # Validate PMT threshold
        threshold = self._config['processing']['pmt_artifact_detection_threshold']
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("pmt_artifact_detection_threshold must be a positive number")
    
    @staticmethod
    def _set_device() -> str:
        """
        Detect available device for TensorFlow operations.
        
        Returns
        -------
        str
            Device string ('/GPU:0' or '/CPU:0').
        """
        return '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    @property
    def processing_params(self) -> dict:
        """Get processing parameters."""
        return self._config['processing'].copy()
    
    @property
    def kernel_size_um(self) -> tuple:
        """Get median filter kernel size."""
        return tuple(self._config['processing']['kernel_size_um'])
    
    @property
    def pmt_artifact_detection_threshold(self) -> float:
        """Get PMT artifact detection threshold."""
        return self._config['processing']['pmt_artifact_detection_threshold']
    
    @property
    def device(self) -> str:
        """Get processing device."""
        return self._device
    
    @property
    def all_params(self) -> dict:
        """Get all configuration parameters."""
        return self._config.copy()
    
