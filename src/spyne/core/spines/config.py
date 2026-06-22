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
        config_path: str | Path = None,
        segmentation_algorithm: str = None,
        deepd3_model_fn: str | Path = None,
        nnunet_model_fn: str | Path = None,
        care_model_fn: str | Path = None,
        classifier_model_fn: str | Path = None,
        **parameter_overrides
    ) -> None:
        """
        Initialize configuration manager.
        
        Parameters
        ----------
        config_path : str | Path, optional
            Path to configuration file. If None, uses default location.
        deepd3_model_fn : str | Path, optional
            Path to DeepD3 segmentation model file. Overrides config value.
        nnunet_model_fn : str | Path, optional
            Path to nnU-Net segmentation model file. Overrides config value.
        care_model_fn : str | Path, optional
            Path to CARE denoising model file. Overrides config value.
        classifier_model_fn : str | Path, optional
            Path to classifier model file. Overrides config value.
        segmentation_algorithm : str, optional
            Segmentation algorithm to use. Overrides config value.
        **parameter_overrides
            Additional parameters to override config values.
        """
        self.repo_root = Path(__file__).parent.parent.parent
        self.config = self._load_config(config_path)
        
        # Resolve model paths
        self.deepd3_model_fn = self._resolve_path(
            deepd3_model_fn or self.config['models']['deepd3_segmentation']['path']
        )
        self.nnunet_model_fn = self._resolve_path(
            nnunet_model_fn or self.config['models']['nnunet_segmentation']['path']
        )
        self.care_model_fn = self._resolve_path(
            care_model_fn or self.config['models']['care_denoising']['model_path']
        )
        self.classifier_model_fn = self._resolve_path(
            classifier_model_fn or self.config['models']['calcium_event_classifier']['path']
        )

        # Set segmentation algorithm (use config default if not provided)
        self.segmentation_algorithm = (
            segmentation_algorithm or
            self.config['segmentation']['segmentation_algorithm']
        )
        self._validate_algorithm(self.segmentation_algorithm)
        
        # Set segmentation parameters with overrides
        self._set_parameters(parameter_overrides)
        
        # Validate model files exist (after _set_parameters so care_enabled
        # reflects any runtime override)
        self._validate_files()
        
        # Detect device for TensorFlow operations
        self.device = self._set_device()
    
    def _load_config(
            self,
            config_path: str | Path = None
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
            config_path = self.repo_root / "config/spine_config.yaml"
        
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
            path: str | Path
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
    
    def _validate_algorithm(self, algorithm: str) -> None:
        """
        Validate that the specified segmentation algorithm is supported.
        
        Parameters
        ----------
        algorithm : str
            Name of the segmentation algorithm to validate.
            
        Raises
        ------
        ValueError
            If the specified algorithm is not supported.
        """
        supported_algorithms = self.config['segmentation']['supported_algorithms']
        
        if algorithm not in supported_algorithms:
            raise ValueError(
                f"Unsupported segmentation algorithm: '{algorithm}'. "
                f"Supported algorithms are: {supported_algorithms}"
            )
    
    def _validate_files(self) -> None:
        """
        Validate that required model files exist.

        Only the segmentation model matching the selected algorithm is checked.
        The CARE model is only checked when denoising is enabled in config.

        Raises
        ------
        FileNotFoundError
            If segmentation or classifier model files don't exist.
        """
        if self.segmentation_algorithm == 'deepd3':
            if not self.deepd3_model_fn.exists():
                raise FileNotFoundError(
                    f'DeepD3 model not found: {self.deepd3_model_fn}'
                )
        elif self.segmentation_algorithm == 'nnunet':
            if not self.nnunet_model_fn.exists():
                raise FileNotFoundError(
                    f'nnU-Net model not found: {self.nnunet_model_fn}'
                )

        if self.care_enabled and not self.care_model_fn.exists():
            raise FileNotFoundError(
                f'CARE denoising model not found: {self.care_model_fn}'
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
        seg_config = self.config.get('segmentation', {})
        deepd3_params = seg_config.get('deepd3_parameters', {})
        nnunet_params = seg_config.get('nnunet_parameters', {})
        post_proc_params = seg_config.get('post_processing_parameters', {})
        ts_params = self.config.get('timeseries', {})
        bg_subtraction_params = self.config.get('background_subtraction', {})
        denoising_config = self.config.get('denoising', {})
        care_params = denoising_config.get('care_parameters', {})

        # Algorithm-specific parameters
        algorithm_params = {
            'spine_threshold': deepd3_params,
            'dendrite_threshold': deepd3_params
        }
        
        # Post-processing parameters (shared across algorithms)
        postproc_param_names = [
            'mask_size', 'min_distance', 'min_spine_size',
            'min_dendrite_size', 'dendrite_dilation_iterations'
        ]
        
        # Set algorithm-specific parameters
        for param_name, source_dict in algorithm_params.items():
            if param_name in overrides and overrides[param_name] is not None:
                setattr(self, param_name, overrides[param_name])
            elif param_name in source_dict:
                setattr(self, param_name, source_dict[param_name])
            else:
                raise KeyError(
                    f"Parameter '{param_name}' not found in config file"
                )
        
        # Set post-processing parameters
        for param_name in postproc_param_names:
            if param_name in overrides and overrides[param_name] is not None:
                setattr(self, param_name, overrides[param_name])
            elif param_name in post_proc_params:
                setattr(self, param_name, post_proc_params[param_name])
            else:
                raise KeyError(
                    f"Parameter '{param_name}' not found in config file"
                )

        # Background subtraction parameters
        bg_subtraction_parameter_names = [
            'background_dilation_disk_size',
            'neuropil_factor',
        ]

        for param_name in bg_subtraction_parameter_names:
            if param_name in overrides and overrides[param_name] is not None:
                setattr(self, param_name, overrides[param_name])
            elif param_name in bg_subtraction_params:
                setattr(self, param_name, bg_subtraction_params[param_name])
            else:
                raise KeyError(
                    f"Parameter '{param_name}' not found in config file"
                )
        
        # Timeseries parameters
        timeseries_parameter_names = [
            'rolling_bsl',
            'window_sec',
            'min_quantile',
            'baseline_percentile',
        ]

        for param_name in timeseries_parameter_names:
            if param_name in overrides and overrides[param_name] is not None:
                setattr(self, param_name, overrides[param_name])
            elif param_name in ts_params:
                setattr(self, param_name, ts_params[param_name])
            else:
                raise KeyError(
                    f"Parameter '{param_name}' not found in config file"
                )
        
        # CARE denoising parameters
        care_parameter_names = [
            'care_enabled',
            'care_python_executable',
            'care_axes',
        ]
        
        # Map config keys to attribute names (handle 'enabled' -> 'care_enabled')
        care_param_mapping = {
            'care_enabled': 'enabled',
            'care_python_executable': 'care_python_executable',
            'care_axes': 'axes',
        }
        
        for param_name in care_parameter_names:
            config_key = care_param_mapping[param_name]
            if param_name in overrides and overrides[param_name] is not None:
                setattr(self, param_name, overrides[param_name])
            elif config_key in care_params:
                setattr(self, param_name, care_params[config_key])
            else:
                raise KeyError(
                    f"Parameter '{config_key}' not found in config file"
                )
        
        # Store supported algorithms for easy access
        self.supported_algorithms = seg_config.get('supported_algorithms', [])
    
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
        # Select model based on current algorithm
        model_fn = (
            self.deepd3_model_fn if self.segmentation_algorithm == 'deepd3'
            else self.nnunet_model_fn
        )
        
        return {
            'supported_algorithms': self.supported_algorithms,
            'segmentation_algorithm': self.segmentation_algorithm,
            'device': self.device,
            'deepd3_model_fn': self.deepd3_model_fn,
            'nnunet_model_fn': self.nnunet_model_fn,
            'segmentation_model_fn': model_fn,  # Current algorithm's model
            'spine_threshold': self.spine_threshold,
            'min_spine_size': self.min_spine_size,
            'mask_size': self.mask_size,
            'min_distance': self.min_distance,
            'dendrite_threshold': self.dendrite_threshold,
            'min_dendrite_size': self.min_dendrite_size,
            'dendrite_dilation_iterations': self.dendrite_dilation_iterations
        }
    
    @property
    def timeseries_params(self) -> dict:
        """Parameters for timeseries extraction."""
        return {
            'background_dilation_disk_size': self.background_dilation_disk_size,
        }

    @property
    def classification_params(self) -> dict:
        """Parameters for calcium event classification."""
        return {
            'classifier_model_fn': self.classifier_model_fn,
        }
    
    @property
    def care_params(self) -> dict:
        """Parameters for CARE denoising."""
        return {
            'care_model_fn': self.care_model_fn,
            'care_enabled': self.care_enabled,
            'care_python_executable': self.care_python_executable,
            'care_axes': self.care_axes,
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