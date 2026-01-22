""" Configuration management for SpineDataset
    Created on January 21, 2026
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
import yaml
import tensorflow as tf


class SpineDatasetConfig:
    """
    Manages configuration loading and parameter validation for spine analysis.
    
    This class handles:
    - YAML configuration file loading
    - Parameter validation and type checking
    - Model path resolution
    - Device detection for TensorFlow operations
    """
    
    def __init__(
        self,
        config_path: str or Path = None,
        segmentation_model_fn: str or Path = None,
        classifier_model_fn: str or Path = None,
        **parameter_overrides
    ) -> None:
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to configuration file. If None, uses default location.
        segmentation_model_fn : str or Path, optional
            Path to segmentation model file. Overrides config value.
        classifier_model_fn : str or Path, optional
            Path to classifier model file. Overrides config value.
        **parameter_overrides
            Additional parameters to override config values.
        """
        self.repo_root = Path(__file__).parent.parent.parent
        self.config = self._load_config(config_path)
        
        # Resolve model paths
        self.segmentation_model_fn = self._resolve_path(
            segmentation_model_fn or self.config['models']['segmentation']['path']
        )
        self.classifier_model_fn = self._resolve_path(
            classifier_model_fn or self.config['models']['classifier']['path']
        )
        
        # Validate model files exist
        self._validate_files()
        
        # Set segmentation parameters with overrides
        self._set_parameters(parameter_overrides)
        
        # Detect device for TensorFlow operations
        self.device = self._set_device()
    
    def _load_config(
            self,
            config_path: str or Path = None
    ) -> dict:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to configuration file.
            
        Returns
        -------
        dict
            Configuration dictionary.
            
        Raises
        ------
        FileNotFoundError
            If config file doesn't exist.
        yaml.YAMLError
            If YAML file is malformed.
        """
        if config_path is None:
            config_path = self.repo_root / "config/spyne_config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Create a config file or ensure it exists at the expected location."
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing configuration file {config_path}: {e}")
    
    def _resolve_path(
            self,
            path: str or Path
    ) -> Path:
        """
        Convert model path to absolute path relative to repo root.
        
        Parameters
        ----------
        model_path : str | Path
            Model path from config or parameter.
            
        Returns
        -------
        Path
            Absolute path to model file.
        """
        path = Path(path)
        
        if path.is_absolute():
            return path
        
        return self.repo_root / path
    
    def _validate_files(self) -> None:
        """
        Validate that required model files exist.
        
        Raises
        ------
        FileNotFoundError
            If segmentation or classifier model files don't exist.
        """
        if not self.segmentation_model_fn.exists():
            raise FileNotFoundError(
                f'Segmentation model not found: {self.segmentation_model_fn}'
            )
        
        if not self.classifier_model_fn.exists():
            raise FileNotFoundError(
                f'Classifier model not found: {self.classifier_model_fn}'
            )
    
    def _set_parameters(
            self,
        overrides: dict
    ) -> None:
        """
        Set segmentation parameters with priority: overrides > config.
        
        Parameters
        ----------
        overrides : dict
            Parameter overrides from constructor arguments.
        """
        seg_params = self.config.get('segmentation', {})
        
        # Define parameter names that can be overridden
        parameter_names = [
            'spine_threshold', 'dendrite_threshold', 'mask_size',
            'min_distance', 'min_spine_size', 'min_dendrite_size',
            'dendrite_dilation_iterations'
        ]
        
        for param_name in parameter_names:
            
            if param_name in overrides and overrides[param_name] is not None:
                setattr(self, param_name, overrides[param_name])
            
            elif param_name in seg_params:
                setattr(self, param_name, seg_params[param_name])
            
            else:
                raise KeyError(
                    f"Parameter '{param_name}' not found in config file"
                )
    
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
    def segmentation_params(self) -> dict:
        """Parameters for semantic segmentation pipeline."""
        return {
            'device': self.device,
            'segmentation_model_fn': self.segmentation_model_fn,
            'spine_threshold': self.spine_threshold,
            'min_spine_size': self.min_spine_size,
            'mask_size': self.mask_size,
            'min_distance': self.min_distance,
            'dendrite_threshold': self.dendrite_threshold,
            'min_dendrite_size': self.min_dendrite_size,
            'dendrite_dilation_iterations': self.dendrite_dilation_iterations
        }
    
    @property
    def classification_params(self) -> dict:
        """Parameters for calcium event classification."""
        return {
            'classifier_model_fn': self.classifier_model_fn,
        }
    
    @property
    def params(self) -> dict:
        """Combined parameters for backward compatibility."""
        return {**self.segmentation_params, **self.classification_params}
    
    def validate_parameters(self) -> None:
        """
        Validate parameter values and types.
        
        Raises
        ------
        ValueError
            If parameter values are invalid.
        TypeError
            If parameter types are incorrect.
        """
        validations = [
            (self.spine_threshold, (int, float), "spine_threshold must be numeric"),
            (self.dendrite_threshold, (int, float), "dendrite_threshold must be numeric"),
            (self.mask_size, int, "mask_size must be an integer"),
            (self.min_distance, int, "min_distance must be an integer"),
            (self.min_spine_size, (int, float), "min_spine_size must be numeric"),
            (self.min_dendrite_size, (int, float), "min_dendrite_size must be numeric"),
            (self.dendrite_dilation_iterations, int, "dendrite_dilation_iterations must be an integer"),
        ]
        
        for value, expected_type, error_msg in validations:
            if not isinstance(value, expected_type):
                raise TypeError(error_msg)
            
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{error_msg.split(' must be')[0]} must be non-negative")