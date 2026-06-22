"""Configuration management for electrophysiology datasets.

Created on February 24, 2026
@author: dcupolillo
"""

from __future__ import annotations
from pathlib import Path
import yaml
import tensorflow as tf


class EphyDatasetConfig:
    """
    Configuration manager for electrophysiology dataset processing.

    Handles all configuration parameters for ABF data loading,
    event detection, and validation. Supports YAML-based configuration
    with parameter override capabilities.
    """

    def __init__(
        self,
        config_path: str | Path = None,
        model_fn: str | Path = None,
        recording_channel: int = None,
        win_size: int = None,
        direction: str = None,
        bla_epoch_idx: int = None,
        ca3_epoch_idx: int = None,
        model_threshold: float = None,
        batch_size: int = None,
        peak_w: float = None,
        rel_prom_cutoff: float = None,
        convolve_window: int = None,
        gradient_convolve_window: int = None,
        resample_to_600: bool = None,
        **overrides
    ):
        """
        Initialize configuration manager.

        Parameters
        ----------
        config_path : str | Path, optional
            Path to configuration YAML file. If None, uses default config.
        model_fn : str | Path, optional
            Path to the miniML event detection model. Overrides config value.
        **overrides
            Additional detection parameter overrides (e.g. channel,
            win_size, bla_epoch_idx).
        """
        self._config = self._load_config(config_path)

        if model_fn is not None:
            self._config['model']['path'] = str(model_fn)

        for key, value in overrides.items():
            if key in self._config['detection']:
                self._config['detection'][key] = value

        self._validate_config()
        self._device = self._set_device()

    def _load_config(
            self,
            config_path: str | Path = None
    ) -> dict:
        """Load configuration from YAML file.

        Parameters
        ----------
        config_path : str | Path, optional
            Path to the YAML configuration file.

        Returns
        -------
        dict
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the YAML file is malformed.
        """
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent /
                "config" / "ephy_config.yaml")

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        FileNotFoundError
            If the model file does not exist.
        """
        model_path = Path(self._config['model']['path'])
        if not model_path.exists():
            raise FileNotFoundError(
                f"miniML model not found: {model_path}")

        detection = self._config['detection']

        if detection['channel'] < 0:
            raise ValueError("channel must be a non-negative integer")

        if detection['win_size'] <= 0:
            raise ValueError("win_size must be a positive integer")

        if detection['direction'] not in ('negative', 'positive'):
            raise ValueError(
                "direction must be 'negative' or 'positive'")

        if not 0.0 < detection['model_threshold'] <= 1.0:
            raise ValueError(
                "model_threshold must be in the range (0, 1]")

    @staticmethod
    def _set_device() -> str:
        """Detect available device for TensorFlow operations.

        Returns
        -------
        str
            Device string ('/GPU:0' or '/CPU:0').
        """
        return '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    @property
    def model_fn(self) -> Path:
        """Path to the miniML event detection model."""
        return Path(self._config['model']['path'])

    @property
    def detection_params(self) -> dict:
        """Get all detection parameters."""
        return self._config['detection'].copy()

    @property
    def channel(self) -> int:
        """Recording channel index."""
        return self._config['detection']['channel']

    @property
    def win_size(self) -> int:
        """miniML sliding window size in samples."""
        return self._config['detection']['win_size']

    @property
    def direction(self) -> str:
        """Event detection direction."""
        return self._config['detection']['direction']

    @property
    def bla_epoch_idx(self) -> int:
        """ABF epoch index for BLA stimulation onset."""
        return self._config['detection']['bla_epoch_idx']

    @property
    def ca3_epoch_idx(self) -> int:
        """ABF epoch index for CA3 stimulation onset."""
        return self._config['detection']['ca3_epoch_idx']

    @property
    def device(self) -> str:
        """Processing device string."""
        return self._device

    @property
    def all_params(self) -> dict:
        """Get full configuration dictionary."""
        return self._config.copy()
