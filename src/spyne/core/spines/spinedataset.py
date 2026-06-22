""" Created on Fri Mar  1 16:20:12 2024
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
import numpy as np
from functools import cache, cached_property
from spyne.core.imaging.imagingdataset import ImagingDataset
from spyne.core.spines.config import SpineDatasetConfig
from spyne.core.spines.dataloader import SpineDataLoader
from spyne.core.spines.pipeline import SpineAnalysisPipeline
from spyne.core.spines.analysis.timeseries.timeseries import (
    dFF, get_timestamps, z_score)


class SpineDataset:
    """
    A manager class for analyzing dendritic imaging datasets.

    This class provides functionalities for:
    - Processing imaging data and performing semantic segmentation of spines
        and dendrites using a neural network.
    - Analyzing calcium dynamics within segmented spines.
    - Detecting calcium events using a built-in classifier.
    - Visualizing spines, dendrites, and related data.

    Example
    -------
    >>> import spyne
    >>> paths = "path/to/your/folder"
    >>> dataset = spyne.ImagingDataset(paths)
    >>> spine_dataset = spyne.SpineDataset(dataset)

    >>> # Run segmentation and analyze the dataset
    >>> spine_dataset.collect_all_data()
    >>> spine_dataset.calcium_events_predictions()

    >>> # Visualize detected spines
    >>> spyne.plot.scatter(spine_dataset.spines_data)
    """

    def __init__(
        self,
        dataset: ImagingDataset,
        config_path: str | Path = None,
        # Algorithm selection and model paths
        segmentation_algorithm: str = None,
        deepd3_model_fn: str | Path = None,
        nnunet_model_fn: str | Path = None,
        # CARE denoising parameters
        care_model_fn: str | Path = None,
        enable_denoising: bool = None,
        # Threshold for deepd3 outputs
        spine_threshold: float = None,
        dendrite_threshold: float = None,
        # Post-processing parameters
        mask_size: int = None,
        min_distance: int = None,
        min_spine_size: float = None,
        min_dendrite_size: float = None,
        dendrite_dilation_iterations: int = None,
        # calcium event classifier
        classifier_model_fn: str | Path = None,
    ) -> None:
        """
        Initialize the SpineDataset class.

        Parameters
        ----------
        dataset : ImagingDataset
            The imaging dataset to process.
        config_path : str | Path, optional
            Path to configuration file. If None, uses default location
            (config/spine_config.yaml). Default is None.
        segmentation_model_fn : str | Path, optional
            Path to the trained model file for segmentation.
            Overrides config file value. Default is None.
        care_model_fn : str | Path, optional
            Path to the trained CARE denoising model file.
            Overrides config file value. Default is None.
        enable_denoising : bool, optional
            Whether to enable CARE denoising before segmentation.
            Overrides ``denoising.care_parameters.enabled`` in the config
            file. Default is None (uses config value).
        spine_threshold : float, optional
            Threshold for spine segmentation.
            Overrides config file value. Default is None.
        dendrite_threshold : float, optional
            Threshold for dendrite segmentation.
            Overrides config file value. Default is None.
        mask_size : int, optional
            Size of the morphological mask.
            Overrides config file value. Default is None.
        min_distance : int, optional
            Minimum distance for spine separation.
            Overrides config file value. Default is None.
        min_spine_size : float, optional
            Minimum size for spines.
            Overrides config file value. Default is None.
        min_dendrite_size : float, optional
            Minimum size for dendrites.
            Overrides config file value. Default is None.
        dendrite_dilation_iterations : int, optional
            Number of dilation iterations for dendrite segmentation.
            Overrides config file value. Default is None.
        classifier_model_fn : str | Path, optional
            Path to the trained model file for calcium event classification.
            Overrides config file value. Default is None.

        Raises
        ------
        TypeError
            If the `dataset` is not an instance of `ImagingDataset`.
        FileNotFoundError
            If the config file or model files do not exist.

        Notes
        -----
        Configuration priority (highest to lowest):
        1. Arguments passed to __init__
        2. Values from config_path YAML file
        3. Default values in config/spine_config.yaml

        Order of operations:
        1. Load configuration
        2. Load pre-computed data
        3. Initialize analysis pipeline
        """

        if not isinstance(dataset, ImagingDataset):
            raise TypeError('Invalid input type for dataset')
        
        self._dataset = dataset
        self.metadata = self._dataset.metadata

        # Initialize configuration manager
        self.config = SpineDatasetConfig(
            config_path=config_path,
            segmentation_algorithm=segmentation_algorithm,
            deepd3_model_fn=deepd3_model_fn,
            nnunet_model_fn=nnunet_model_fn,
            care_model_fn=care_model_fn,
            classifier_model_fn=classifier_model_fn,
            care_enabled=enable_denoising,
            spine_threshold=spine_threshold,
            dendrite_threshold=dendrite_threshold,
            mask_size=mask_size,
            min_distance=min_distance,
            min_spine_size=min_spine_size,
            min_dendrite_size=min_dendrite_size,
            dendrite_dilation_iterations=dendrite_dilation_iterations
        )

        # Initialize algorithm-aware storage
        self._segmentation_results = {}
        self._timeseries_results = {}
        self._calcium_events_results = {}
        self._current_algorithm = (
            f"{self.config.segmentation_algorithm}_care"
            if self.config.care_enabled
            else self.config.segmentation_algorithm
        )
        
        # Initialize data loader manager and load pre-computed data
        self.data_loader = SpineDataLoader(self._dataset.folder)
        self._load_data()
        
        # Initialize analysis pipeline
        self._pipeline = None

    @property
    def segmentation_params(self) -> dict:
        """Parameters for semantic segmentation pipeline."""
        return self.config.segmentation_params

    @property
    def classification_params(self) -> dict:
        """Parameters for calcium event classification."""
        return self.config.classification_params

    @property
    def params(self) -> dict:
        """Combined parameters for backward compatibility."""
        return self.config.params
    
    @property
    def pipeline(self) -> SpineAnalysisPipeline:
        """Analysis pipeline for spine processing workflows."""
        if self._pipeline is None:
            self._pipeline = SpineAnalysisPipeline(
                dataset=self._dataset,
                config=self.config,
                data_loader=self.data_loader
            )
        return self._pipeline
    
    @property
    def spines_data(self) -> list:
        """Processed spine data from semantic segmentation."""
        if self._current_algorithm in self._segmentation_results:
            return self._segmentation_results[self._current_algorithm].get('spines_data', [])
        return getattr(self, '_spines_data', [])
    
    @property
    def dendrites_data(self) -> list:
        """Processed dendrite data from semantic segmentation."""
        if self._current_algorithm in self._segmentation_results:
            return self._segmentation_results[self._current_algorithm].get('dendrites_data', [])
        return getattr(self, '_dendrites_data', [])
    
    @property
    def spine_predictions(self) -> list:
        """Raw neural network predictions for spine segmentation."""
        if self._current_algorithm in self._segmentation_results:
            return self._segmentation_results[self._current_algorithm].get('spine_predictions', [])
        return getattr(self, '_spine_predictions', [])
    
    @property
    def dendrite_predictions(self) -> list:
        """Raw neural network predictions for dendrite segmentation."""
        if self._current_algorithm in self._segmentation_results:
            return self._segmentation_results[self._current_algorithm].get('dendrite_predictions', [])
        return getattr(self, '_dendrite_predictions', [])
    
    @property
    def dendrites_masks(self) -> list:
        """Binary masks for dendrites."""
        if self._current_algorithm in self._segmentation_results:
            return self._segmentation_results[self._current_algorithm].get('dendrites_masks', [])
        return getattr(self, '_dendrites_masks', [])
    
    @property
    def combined_spines_masks(self) -> list:
        """Combined binary masks for all spines."""
        if self._current_algorithm in self._segmentation_results:
            return self._segmentation_results[self._current_algorithm].get('combined_spines_masks', [])
        return getattr(self, '_combined_spines_masks', [])
    
    @property
    def zscores_CA3(self) -> list:
        """Z-scored calcium traces for CA3 spines."""
        if self._current_algorithm in self._timeseries_results:
            return self._timeseries_results[self._current_algorithm].get('zscores_CA3', [])
        return getattr(self, '_zscores_CA3', [])
    
    @property
    def dFF_CA3(self) -> list:
        """dF/F traces for CA3 spines."""
        if self._current_algorithm in self._timeseries_results:
            return self._timeseries_results[self._current_algorithm].get('dFF_CA3', [])
        return getattr(self, '_dFF_CA3', [])
    
    @property
    def ts_CA3(self) -> list:
        """Timestamps for CA3 spines."""
        if self._current_algorithm in self._timeseries_results:
            return self._timeseries_results[self._current_algorithm].get('ts_CA3', [])
        return getattr(self, '_ts_CA3', [])
    
    @property
    def zscores_BLA(self) -> list:
        """Z-scored calcium traces for BLA spines."""
        if self._current_algorithm in self._timeseries_results:
            return self._timeseries_results[self._current_algorithm].get('zscores_BLA', [])
        return getattr(self, '_zscores_BLA', [])
    
    @property
    def dFF_BLA(self) -> list:
        """dF/F traces for BLA spines."""
        if self._current_algorithm in self._timeseries_results:
            return self._timeseries_results[self._current_algorithm].get('dFF_BLA', [])
        return getattr(self, '_dFF_BLA', [])
    
    @property
    def ts_BLA(self) -> list:
        """Timestamps for BLA spines."""
        if self._current_algorithm in self._timeseries_results:
            return self._timeseries_results[self._current_algorithm].get('ts_BLA', [])
        return getattr(self, '_ts_BLA', [])
    
    @property
    def calcium_events_probabilities_BLA(self) -> list:
        """Calcium event probabilities for BLA spines."""
        if self._current_algorithm in self._calcium_events_results:
            return self._calcium_events_results[self._current_algorithm].get('calcium_event_probabilities_BLA', [])
        return getattr(self, '_calcium_events_probabilities_BLA', [])
    
    @property
    def calcium_events_probabilities_CA3(self) -> list:
        """Calcium event probabilities for CA3 spines."""
        if self._current_algorithm in self._calcium_events_results:
            return self._calcium_events_results[self._current_algorithm].get('calcium_event_probabilities_CA3', [])
        return getattr(self, '_calcium_events_probabilities_CA3', [])

    @property
    def denoised_images(self) -> list | np.ndarray | None:
        """CARE-denoised images, unwrapped from care_denoised_images.h5."""
        raw = getattr(self, '_care_denoised_images', None)
        if isinstance(raw, dict) and 'denoised_images' in raw:
            imgs = raw['denoised_images']
            if isinstance(imgs, dict):
                # flammkuchen stored ragged arrays as numbered sub-groups
                return [imgs[str(i)] for i in range(len(imgs))]
            return imgs
        return raw

    def set_algorithm(self, algorithm: str) -> None:
        """
        Switch to a different algorithm's results.

        Parameters
        ----------
        algorithm : str
            Name of the segmentation algorithm ('deepd3' or 'nnunet').

        Raises
        ------
        ValueError
            If no results exist for the specified algorithm.

        Notes
        -----
        - Switches which algorithm's results are accessed via properties.
        - Updates n_spines and clears cached properties for consistency.
        
        Example
        -------
        >>> spine_dataset.collect_spines_and_dendrites_data(algorithm='deepd3')
        >>> spine_dataset.collect_spines_and_dendrites_data(algorithm='nnunet')
        >>> spine_dataset.set_algorithm('deepd3')  # Switch back to DeepD3
        >>> print(len(spine_dataset.spines_data))  # Now shows DeepD3 count
        """

        if algorithm not in self._segmentation_results:
            available = list(self._segmentation_results.keys())
            raise ValueError(
                f"No results for algorithm '{algorithm}'. "
                f"Available: {available}. "
                f"Run collect_spines_and_dendrites_data(algorithm='{algorithm}') first.")

        self._current_algorithm = algorithm
        self.n_spines = len(self.spines_data)
        self._clear_cached_properties()
        
        # Clear ROI cache to force re-creation with new algorithm data
        if hasattr(self._get_roi, 'cache_clear'):
            self._get_roi.cache_clear()
    
    def available_algorithms(self) -> list[str]:
        """
        Get list of algorithms with stored results.
        
        Returns
        -------
        list[str]
            List of algorithm names that have been run.
        
        Example
        -------
        >>> spine_dataset.collect_spines_and_dendrites_data(algorithm='deepd3')
        >>> spine_dataset.available_algorithms()
        ['deepd3']
        >>> spine_dataset.collect_spines_and_dendrites_data(algorithm='nnunet')
        >>> spine_dataset.available_algorithms()
        ['deepd3', 'nnunet']
        """
        return list(self._segmentation_results.keys())
    
    def get_n_spines(self, algorithm: str = None) -> int:
        """
        Get number of spines for a specific algorithm.
        
        Parameters
        ----------
        algorithm : str, optional
            Algorithm name ('deepd3' or 'nnunet').
            If None, uses current algorithm.
        
        Returns
        -------
        int
            Number of detected spines for the specified algorithm.
        
        Example
        -------
        >>> n_deepd3 = spine_dataset.get_n_spines('deepd3')
        >>> n_nnunet = spine_dataset.get_n_spines('nnunet')
        """
        algorithm = (
            algorithm if algorithm is not None
            else self._current_algorithm
        )
        if algorithm in self._segmentation_results:
            spines = self._segmentation_results[algorithm].get(
                'spines_data', [])
            return len(spines)
        return 0
    
    def get_algorithm_data(
            self,
            algorithm: str,
            data_type: str
    ) -> list | np.ndarray:
        """
        Get specific data for an algorithm without changing current algorithm.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name ('deepd3' or 'nnunet').
        data_type : str
            Type of data to retrieve (e.g., 'spines_data', 'zscores_CA3').
        
        Returns
        -------
        list | np.ndarray
            Requested data, or empty list if not found.
        
        Example
        -------
        >>> spines = spine_dataset.get_algorithm_data('deepd3', 'spines_data')
        >>> zscores = spine_dataset.get_algorithm_data('nnunet', 'zscores_CA3')
        """
        # Check all result categories
        for results_dict in [
            self._segmentation_results,
            self._timeseries_results,
            self._calcium_events_results
        ]:
            if algorithm in results_dict:
                if data_type in results_dict[algorithm]:
                    return results_dict[algorithm][data_type]
        return []

    def _load_data(self) -> None:
        """
        Load precomputed data (if available) for faster analysis.
        Algorithm-aware loading: detects algorithm from filename suffix.
        
        Organizes data into three categories:
        - Segmentation results (spines_data, dendrites_data, etc.)
        - Timeseries results (zscores, dFF, timestamps)
        - Calcium event results (event probabilities)
        """
        # Define which attributes belong to each category
        SEGMENTATION_ATTRS = {
            'spines_data', 'dendrites_data', 'spine_predictions',
            'dendrite_predictions', 'dendrites_masks', 'combined_spines_masks'
        }
        TIMESERIES_ATTRS = {
            'zscores_CA3', 'dFF_CA3', 'ts_CA3',
            'zscores_BLA', 'dFF_BLA', 'ts_BLA'
        }
        CALCIUM_EVENTS_ATTRS = {
            'calcium_events_probabilities_BLA',
            'calcium_events_probabilities_CA3'
        }

        loaded_data = self.data_loader.load_multiple_files()
        
        for attr_name, data in loaded_data.items():
            # Skip empty placeholders (files that didn't exist on disk)
            if isinstance(data, list) and len(data) == 0:
                continue

            # Check if filename has algorithm suffix (e.g., spines_data_deepd3)
            algorithm = self._extract_algorithm_from_attr(attr_name)
            
            if algorithm:
                # Store in algorithm-specific dict
                base_name = attr_name.rsplit(f'_{algorithm}', 1)[0]
                
                # Determine which category this attribute belongs to
                if base_name in SEGMENTATION_ATTRS:
                    if algorithm not in self._segmentation_results:
                        self._segmentation_results[algorithm] = {}
                    self._segmentation_results[algorithm][base_name] = data
                
                elif base_name in TIMESERIES_ATTRS:
                    if algorithm not in self._timeseries_results:
                        self._timeseries_results[algorithm] = {}
                    self._timeseries_results[algorithm][base_name] = data
                
                elif base_name in CALCIUM_EVENTS_ATTRS:
                    if algorithm not in self._calcium_events_results:
                        self._calcium_events_results[algorithm] = {}
                    self._calcium_events_results[algorithm][base_name] = data
            else:
                # Legacy files without algorithm suffix - store as private attribute
                setattr(self, f'_{attr_name}', data)

        # Keep the algorithm from config; do not override with loaded data
        # (loaded data is stored correctly in _segmentation_results regardless)

        self.n_spines = len(self.spines_data)
        self._clear_cached_properties()
        if hasattr(self._get_roi, 'cache_clear'):
            self._get_roi.cache_clear()

    def _extract_algorithm_from_attr(self, attr_name: str) -> str | None:
        """
        Extract algorithm name from attribute.
        Useful for inferring which algorithm's results are stored
        in a given attribute based on filename suffix.
        
        Parameters
        ----------
        attr_name : str
            Attribute name (e.g., 'spines_data_deepd3').
        
        Returns
        -------
        str | None
            Algorithm name if found, None otherwise.
        
        Example
        -------
        >>> self._extract_algorithm_from_attr('spines_data_deepd3')
        'deepd3'
        >>> self._extract_algorithm_from_attr('spines_data')
        None
        """
        for algo in self.config.supported_algorithms:
            if attr_name.endswith(f'_{algo}_care'):
                return f'{algo}_care'
            if attr_name.endswith(f'_{algo}'):
                return algo
        return None

    def _clear_cached_properties(self) -> None:
        """
        Clear cached properties that depend on spines_data.
        
        This should be called whenever spines_data is modified
        to ensure cached lookups reflect the current data.
        """
        # Remove cached property values to force recomputation
        for prop_name in ['_spine_indices_by_roi',
                          '_spines_by_branch',
                          '_spines_by_branch_degree']:
            if prop_name in self.__dict__:
                delattr(self, prop_name)

    def load_file(
            self,
            filepath: str | Path
    ) -> tuple:
        """
        User-oriented method to load a single .h5 file and set it
        as an instance attribute. This is a public method that allows
        loading individual data files by pathname.
        The attribute name is inferred from the filename.

        Parameters
        ----------
        filepath : str | Path
            Path to the .h5 file to load.

        Returns
        -------
        tuple
            Tuple containing (attribute_name, loaded_data).

        Example
        -------
        >>> spine_dataset.load_file('path/to/spines_data.h5')
        ('spines_data', [...])  # Returns attribute name and loaded data

        >>> spine_dataset.load_file('path/to/custom_analysis.h5')
        ('custom_analysis', [...])  # Custom files get stem as attribute name
        """
        attr_name, data = self.data_loader.load_file(filepath)
        setattr(self, attr_name, data)
        
        return attr_name, data

    def collect_all_data(
            self,
            save: bool = True,
            save_path: str | Path = None,
            segmentation_algorithm: str = None
    ) -> None:
        """
        Collect and process spine and dendrite segmentation data
        and collects within-spine time series
        for the entire dataset.
        
        This method performs the following steps:
        1. Runs the semantic segmentation pipeline for all ROIs,
           storing the results in `spines_data` and `dendrites_data`.
        2. Collects timeseries data (z-scores, dF/F, timestamps)
           for all segmented spines.
        3. Detects calcium events in spines using a trained neural
           network classifier.

        Alternatively, users can run each step separately using the more specific
        methods `collect_spines_and_dendrites_data()`, `collect_timeseries()`, and
        `calcium_events_predictions()` if they want more control over the workflow
        or want to run different algorithms for each step.
        
        Parameters
        ----------
        save : bool
            Set to True to save data in `.h5` format. Default is True.
        save_path : str or Path, optional
            Directory where the `.h5` files will be saved.
            If None, uses the dataset's parent folder.
            
        Attributes Updated
        ------------------
        spines_data : list
            Processed spine data for all ROIs.
        dendrites_data : list
            Processed dendrite data for all ROIs.
        spine_predictions : list
            Raw neural network predictions for spine segmentation.
        dendrite_predictions : list
            Raw neural network predictions for dendrite segmentation.
        zscores_CA3, dFF_CA3, ts_CA3 : np.ndarray
            Time series data for CA3 spines.
        zscores_BLA, dFF_BLA, ts_BLA : np.ndarray
            Time series data for BLA spines.
        calcium_event_probabilities_BLA, calcium_event_probabilities_CA3 : list
            Calcium event detection results.
            
        Notes
        -----
        - Uses SpineAnalysisPipeline for organized workflow execution.
        - Maintains backward compatibility by setting instance attributes.
        """

        results = self.pipeline.run_pipeline(
            spine_dataset=self,
            save=save,
            save_path=save_path,
            segmentation_algorithm=segmentation_algorithm)
        
        for attr_name, data in results.items():
            setattr(self, attr_name, data)
        
        self.n_spines = len(getattr(self, 'spines_data', []))

        # Invalidate cached properties since spines_data changed
        self._clear_cached_properties()
        if hasattr(self._get_roi, 'cache_clear'):
            self._get_roi.cache_clear()
        
        # Store spine datasets for ROI access
        self._spine_datasets = self.pipeline.get_spine_datasets()

    def apply_denoising(
            self,
            save: bool,
            save_path: str | Path = None,
            overwrite: bool = False
    ) -> None:
        """
        Apply image denoising preprocessing before segmentation.

        Currently uses CARE (Content-Aware Image Restoration) via CSBDeep.
        Future versions may support additional denoising methods.

        Parameters
        ----------
        save : bool, optional
            Whether to save denoised images. Default is True.
        save_path : str | Path, optional
            Directory where result files will be saved.
            If None, uses paths from data_loader mappings.
        overwrite : bool, optional
            Whether to overwrite existing denoised images.
            Default is False.

        Raises
        ------
        ValueError
            If denoising is not enabled in configuration.

        Notes
        -----
        - Saves denoised images as h5 file for programmatic access.
        - Output path: processed/imaging/care_denoised_images.h5
        - Temporary folders (padded inputs, CARE outputs) are deleted
          after the h5 is written.
        - Uses SpineAnalysisPipeline.run_denoising_pipeline() for
          organized workflow execution.

        Examples
        --------
        >>> spine_dataset.apply_denoising()
        >>> spine_dataset.denoised_images  # access results
        """
        denoised = self.pipeline.run_denoising_pipeline(
            spine_dataset=self,
            save=save,
            save_path=save_path,
            overwrite=overwrite
        )
        self._care_denoised_images = denoised

    def collect_spines_and_dendrites_data(
            self,
            save: bool,
            save_path: str | Path = None,
            segmentation_algorithm: str = None
    ) -> list:
        """
        Collect and process spine and dendrite segmentation
        data for the entire dataset.

        This method performs the following steps:
        1. Runs the semantic segmentation pipeline for all ROIs,
        storing the results in `spines_data` and `dendrites_data`
        and `spines_predictions` and `dendrites_predictions`.
        2. Associates segmented spines with their corresponding ROI
        segmenters for indexing.

        Parameters
        ----------
        save : bool
            Set to True to save data in .h5 format. Default is True.
        save_path : str | Path, optional
            Directory where result files will be saved.
            If None, uses the dataset folder.
        segmentation_algorithm : str, optional
            Segmentation algorithm to use ('deepd3' or 'nnunet').
            If None, uses value from config.

        Attributes Updated
        ------------------
        spines_data : list
            Processed spine data for all ROIs.
        dendrites_data : list
            Processed dendrite data for all ROIs.
        spine_predictions : list
            Raw neural network predictions for spine segmentation.
        dendrite_predictions : list
            Raw neural network predictions for dendrite segmentation.

        Notes
        -----
        - Uses SpineAnalysisPipeline for organized segmentation workflow.
        - Results are accessible via properties: spines_data, dendrites_data, etc.
        - Maintains backward compatibility by setting instance attributes.
        """

        segmentation_algorithm = (
            segmentation_algorithm if segmentation_algorithm is not None
            else self.config.segmentation_algorithm
        )

        effective_algorithm = (
            f"{segmentation_algorithm}_care"
            if self.config.care_enabled
            else segmentation_algorithm
        )

        # Auto-run denoising if enabled and not yet done
        if self.config.care_enabled and self.denoised_images is None:
            print("CARE denoising enabled — running apply_denoising() "
                  "before segmentation...")
            self.apply_denoising(save=True)

        # Warn if overwriting existing results for this algorithm
        if effective_algorithm in self._segmentation_results:
            import warnings
            warnings.warn(
                f"Overwriting existing {effective_algorithm} results. "
                f"Previous {effective_algorithm} data will be lost.",
                UserWarning,
                stacklevel=2
            )

        # Run segmentation pipeline
        results = self.pipeline.run_segmentation_pipeline(
            spine_dataset=self,
            save=save,
            save_path=save_path,
            algorithm=effective_algorithm
        )

        # Store in algorithm-specific dictionary
        self._segmentation_results[effective_algorithm] = results
        self._current_algorithm = effective_algorithm
        
        # Also set as private attributes for backward compatibility
        for attr_name, data in results.items():
            setattr(self, f'_{attr_name}', data)

        self.n_spines = len(self.spines_data)
        self._spine_datasets = self.pipeline.get_spine_datasets()
        
        # Invalidate cached properties since spines_data changed
        self._clear_cached_properties()
        
        print(f"Segmentation complete: {self.n_spines} spines detected across {len(self)} ROIs")

    def collect_timeseries(
            self,
            save: bool,
            save_path: str | Path = None,
            segmentation_algorithm: str = None
    ) -> None:
        """
        Collect timeseries data (z-scores, dF/F, timestamps) for all spines.
        This method processes the segmented spines obtained with either deepd3
        or nnunet to extract relevant information, including z-scores,
        dF/F values, and timestamps.

        Parameters
        ----------
        save : bool, optional
            Set to True to save data in .h5 format. Default is True.
        save_path : str | Path, optional
            Directory where the .h5 files will be saved.
            If None, uses the dataset's parent folder.
        segmentation_algorithm : str, optional
            Algorithm's spine mask results to consider wherefrom getting timeseries.
            If None, uses current algorithm.

        Raises
        ------
        ValueError
            If spine data is not available (run segmentation first).
            
        Notes
        -----
        - Uses SpineAnalysisPipeline for organized timeseries collection.
        - Maintains backward compatibility by setting instance attributes.
        - Timeseries are stored per algorithm to match spine segmentation.
        """
        segmentation_algorithm = segmentation_algorithm if segmentation_algorithm is not None else self._current_algorithm
        
        # Fetch algorithm-specific data (not current algorithm's data)
        if segmentation_algorithm not in self._segmentation_results:
            raise ValueError(
                f"No segmentation results for algorithm '{segmentation_algorithm}'. "
                f"Available: {list(self._segmentation_results.keys())}. "
                f"Run collect_spines_and_dendrites_data(algorithm='{segmentation_algorithm}') first.")
        
        algo_results = self._segmentation_results[segmentation_algorithm]
        spines_data = algo_results.get('spines_data', [])
        dendrites_masks = algo_results.get('dendrites_masks', [])
        
        if not spines_data:
            raise ValueError(
                f"No spine data for algorithm '{segmentation_algorithm}'.")

        # Run timeseries pipeline
        print(f"Collecting time series data for {segmentation_algorithm} spines...")
        results = self.pipeline.run_timeseries_pipeline(
            spines_data=spines_data,
            dendrites_masks=dendrites_masks,
            save=save,
            save_path=save_path,
            algorithm=segmentation_algorithm
        )
        
        # Store in algorithm-specific dictionary
        if segmentation_algorithm not in self._timeseries_results:
            self._timeseries_results[segmentation_algorithm] = {}
        self._timeseries_results[segmentation_algorithm].update(results)

    def calcium_events_predictions(
            self,
            save: bool,
            save_path: str | Path = None,
            segmentation_algorithm: str = None
    ) -> None:
        """
        Detect calcium events in spines using a trained neural network
        classifier.

        This method processes traces for BLA and CA3 spines to
        detect calcium events using a pre-trained neural network classifier.
        The results are stored as probabilities values.

        Parameters
        ----------
        save : bool, optional
            Whether to save the calcium event data to .h5 files.
            Default is True.
        save_path : str | Path, optional
            Directory where the .h5 files will be saved.
            If None, uses the dataset's parent folder.
        segmentation_algorithm : str, optional
            Algorithm to use for calcium event detection.
            If None, uses current algorithm.

        Attributes Updated
        ------------------
        calcium_event_probabilities_BLA : list
            Raw calcium event probabilities for BLA spines.
        calcium_event_probabilities_CA3 : list
            Raw calcium event probabilities for CA3 spines.

        Notes
        -----
        - Uses SpineAnalysisPipeline for organized calcium event detection.
        - Maintains backward compatibility by setting instance attributes.
        - Events are stored per algorithm to match timeseries data.

        Raises
        ------
        ValueError
            If time series data is not available.
        """
        segmentation_algorithm = segmentation_algorithm if segmentation_algorithm is not None else self._current_algorithm
        
        # Fetch algorithm-specific timeseries data (not current algorithm's data)
        if segmentation_algorithm not in self._timeseries_results:
            raise ValueError(
                f"No timeseries results for algorithm '{segmentation_algorithm}'. "
                f"Available: {list(self._timeseries_results.keys())}. "
                f"Run collect_timeseries(algorithm='{segmentation_algorithm}') first.")
        
        algo_timeseries = self._timeseries_results[segmentation_algorithm]
        
        # Prepare timeseries data for pipeline
        timeseries_data = {
            'zscores_BLA': algo_timeseries.get('zscores_BLA', []),
            'dFF_BLA': algo_timeseries.get('dFF_BLA', []),
            'zscores_CA3': algo_timeseries.get('zscores_CA3', []),
            'dFF_CA3': algo_timeseries.get('dFF_CA3', [])
        }
        
        # Validate data availability
        if (len(timeseries_data['zscores_BLA']) == 0 and
                len(timeseries_data['zscores_CA3']) == 0):
            raise ValueError(
                f"No timeseries data for algorithm '{segmentation_algorithm}'.")
        
        # Run calcium events pipeline
        print(f"Detecting calcium events for {segmentation_algorithm} spines...")
        results = self.pipeline.run_calcium_events_pipeline(
            timeseries_data=timeseries_data,
            save=save,
            save_path=save_path,
            algorithm=segmentation_algorithm
        )
        
        # Store in algorithm-specific dictionary
        if segmentation_algorithm not in self._calcium_events_results:
            self._calcium_events_results[segmentation_algorithm] = {}
        self._calcium_events_results[segmentation_algorithm].update(results)

    @cached_property
    def _spine_indices_by_roi(self) -> dict:
        """
        Cache spine indices grouped by ROI for efficient lookup.
        
        Returns
        -------
        dict
            Dictionary mapping ROI indices to lists of spine indices.
            
        Example: 
            {0: [0, 1, 2], 1: [3, 4], ...}
            where keys are ROI indices and values are lists of spine indices.
        """
        indices_by_roi = {}
        
        for n, spine in enumerate(self.spines_data):
            roi_n = spine['roi_n']
            if roi_n not in indices_by_roi:
                indices_by_roi[roi_n] = []
            indices_by_roi[roi_n].append(n)
        
        return indices_by_roi
    
    @cached_property
    def _spines_by_branch(self) -> dict:
        """
        Cache spines grouped by branch ID for efficient lookup.
        
        Returns
        -------
        dict
            Dictionary mapping branch IDs to lists of spines.
        """
        spines_by_branch = {}
        
        for spine in self.spines_data:
            branch_id = spine['branch_id']
            if branch_id not in spines_by_branch:
                spines_by_branch[branch_id] = []
            spines_by_branch[branch_id].append(spine)
        
        return spines_by_branch
    
    @cached_property
    def _spines_by_branch_degree(self) -> dict:
        """
        Cache spines grouped by branch degree for efficient lookup.
        
        Returns
        -------
        dict
            Dictionary mapping branch degrees to lists of spines.
        """
        spines_by_degree = {}
        
        for spine in self.spines_data:
            branch_degree = spine['branch_degree']
            if branch_degree not in spines_by_degree:
                spines_by_degree[branch_degree] = []
            spines_by_degree[branch_degree].append(spine)
        
        return spines_by_degree
    
    def spines_by_roi(
            self,
            roi_index: int,
    ) -> list:
        """
        Retrieve spines associated with a specific ROI index.

        Parameters
        ----------
        roi_index : int
            ROI index to filter spines.

        Returns
        -------
        list
            List of spines associated with the specified ROI index.
            Sublist of self.spines_data.
        """
        if not isinstance(roi_index, int):
            raise TypeError("roi_index must be an integer")
        
        if roi_index < 0:
            raise ValueError("roi_index must be non-negative")

        if roi_index >= len(self._dataset):
            raise IndexError("Roi index out of range.")

        spine_indices = self._spine_indices_by_roi.get(roi_index, [])

        if not spine_indices:
            raise ValueError("Selected ROI has no detected spines.")

        return self._fetch_spine_data(self.spines_data, spine_indices)

    def spines_by_branch(
            self,
            branch_id: int,
    ) -> list:
        """
        Retrieve spines associated with a specific branch ID.

        Parameters
        ----------
        branch_id : int
            Branch ID to filter spines.

        Returns
        -------
        list
            List of spines associated with the specified branch ID.
            Sublist of self.spines_data.
        """
        if not isinstance(branch_id, int):
            raise TypeError("branch_id must be an integer")
        
        if branch_id < 0:
            raise ValueError("branch_id must be non-negative")

        if branch_id >= self._dataset.sf.n_branches:
            raise IndexError("Branch id out of range.")

        spines = self._spines_by_branch.get(branch_id, [])

        if not spines:
            raise ValueError("Selected branch has no detected spines.")

        return spines

    def spines_by_branch_degree(self, branch_degree: int) -> list:
        """
        Retrieve spines associated with a specific branch degree.

        Parameters
        ----------
        branch_degree : int
            Branch degree to filter spines.

        Returns
        -------
        list
            List of spines associated with the specified branch degree.
            Sublist of self.spines_data.
        """
        if not isinstance(branch_degree, int):
            raise TypeError("branch_degree must be an integer")
        
        if branch_degree < 0:
            raise ValueError("branch_degree must be non-negative")
        
        if branch_degree not in self._spines_by_branch_degree:
            raise IndexError("Branch degree not found in dataset.")

        return self._spines_by_branch_degree[branch_degree]

    def _fetch_spine_data(
            self,
            data_batch: list | np.ndarray,
            spine_indices: list
    ) -> np.ndarray:
        """
        Helper function to retrieve data from a given ROI based on
        spine indices.

        Parameters
        ----------
        data_batch : list | np.ndarray
            Batch of spine-related data.
        spine_indices : list[int]
            List of indices corresponding to the spines of the given ROI.

        Returns
        -------
        list or np.ndarray
            Filtered data corresponding to the given ROI.
        """

        if data_batch is None or len(data_batch) == 0:
            return np.array([])
        
        return np.array(data_batch)[spine_indices]

    @cache
    def _get_roi(self, roi_index: int) -> RoiSpine:
        """
        Retrieve a specific ROI's segmentation and associated data.

        Parameters
        ----------
        roi_index : int
            Index of the ROI to retrieve.

        Returns
        -------
        RoiSpine
                An instance containing individual ROI data, metadata, and
                segmentation.

        Raises
        ------
        IndexError
            If the specified `roi_index` is out of bounds.
        """

        roi_metadata = self.metadata[roi_index]

        # Use cached spine indices for efficient lookup
        spine_indices = self._spine_indices_by_roi.get(roi_index, [])

        selected_dFF_CA3 = self._fetch_spine_data(
            self.dFF_CA3, spine_indices)
        selected_zscore_CA3 = self._fetch_spine_data(
            self.zscores_CA3, spine_indices)
        selected_ts_CA3 = self._fetch_spine_data(
            self.ts_CA3, spine_indices)
        selected_dFF_BLA = self._fetch_spine_data(
            self.dFF_BLA, spine_indices)
        selected_zscore_BLA = self._fetch_spine_data(
            self.zscores_BLA, spine_indices)
        selected_ts_BLA = self._fetch_spine_data(
            self.ts_BLA, spine_indices)
        selected_calcium_events_BLA = [
            prob for n, spine in enumerate(self.calcium_events_probabilities_BLA)
            for prob in spine if n in spine_indices]
        selected_calcium_events_CA3 = [
            prob for n, spine in enumerate(self.calcium_events_probabilities_CA3)
            for prob in spine if n in spine_indices]
        
        dendrites_masks = self.dendrites_masks
        combined_spines_masks = self.combined_spines_masks
        selected_dendrites_mask = (
            dendrites_masks[roi_index]
            if dendrites_masks and len(dendrites_masks) > roi_index else None
        )
        selected_combined_spines_mask = (
            combined_spines_masks[roi_index]
            if combined_spines_masks and len(combined_spines_masks) > roi_index else None
        )

        selected_spines_data = [
            self.spines_data[i] for i in spine_indices]

        # Handle case where predictions haven't been generated yet
        spine_predictions = (
            self.spine_predictions[roi_index]
            if len(self.spine_predictions) > roi_index
            else None
        )
        dendrite_predictions = (
            self.dendrite_predictions[roi_index]
            if len(self.dendrite_predictions) > roi_index
            else None
        )

        denoised_image = (
            self.denoised_images[roi_index]
            if self.config.care_enabled
            and self.denoised_images is not None
            and len(self.denoised_images) > roi_index
            else None
        )

        algorithm = self._current_algorithm

        return RoiSpine(
                roi_index,
                roi_metadata,
                self._dataset[roi_index],
                self.params,
                selected_dFF_CA3,
                selected_zscore_CA3,
                selected_ts_CA3,
                selected_dFF_BLA,
                selected_zscore_BLA,
                selected_ts_BLA,
                selected_spines_data,
                selected_calcium_events_BLA,
                selected_calcium_events_CA3,
                spine_predictions,
                dendrite_predictions,
                selected_dendrites_mask,
                selected_combined_spines_mask,
                algorithm,
                denoised_image)

    def __getitem__(self, roi_index: int) -> RoiSpine:
        if roi_index not in self._dataset.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self._dataset.roi_list)}')
        return self._get_roi(roi_index)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self) -> RoiSpine:
        if self._current_index < len(self._dataset):
            roi_spine = self._get_roi(self._current_index)
            self._current_index += 1
            return roi_spine
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._dataset)
    

class RoiSpine:
    """
    Manage segmentation and analysis of spines and dendrites
    for a specific ROI.

    This class provides methods for:
    - Accessing and iterating over segmented spines.
    - Visualizing segmentation results, dF/F traces, and z-scores.

     Example
    -------
    >>> import spyne
    >>> from neuronpath.path import neuronpath
    >>> paths = neuronpath('YYMMDD', cell_number)
    >>> dataset = spyne.ImagingDataset(paths)
    >>> segmenter = spyne.DatasetSegmenter(dataset)

    >>> # Run segmentation and analyze the dataset
    >>> segmenter.collect_all_data()

    >>> # Access a specific ROI
    >>> segmented_roi = segmenter[0]
    >>> # Display analysis results
    >>> segmented_roi.plot_masks()
    >>> segmented_roi.plot_dFF()
    >>> segmented_roi.plot_zscore()
    """

    def __init__(
            self,
            roi_index: int,
            roi_metadata: dict,
            roi_data: np.ndarray,
            params: dict,
            dFF_CA3: list,
            zscore_CA3: list,
            ts_CA3: list,
            dFF_BLA: list,
            zscore_BLA: list,
            ts_BLA: list,
            spines_data: list,
            calcium_events_BLA: list,
            calcium_events_CA3: list,
            spine_predictions: np.ndarray,
            dendrite_predictions: np.ndarray,
            dendrites_mask: np.ndarray,
            combined_spines_mask: np.ndarray,
            algorithm: str,
            denoised_image: np.ndarray = None
    ) -> None:
        """
    Initialize the RoiSpine instance for a specific ROI.

        Parameters
        ----------
        roi_index : int
            Index of the region of interest (ROI) within the dataset.
        roi_metadata : dict
            Metadata for the ROI, including information about spatial
            properties, imaging parameters, and number of sweeps.
        roi_data : np.ndarray
            Raw ROI data containing imaging or fluorescence frames.
        params : dict
            Configuration parameters for segmentation, including thresholds and
            processing settings.
        dFF_CA3 : list
            dF/F traces for spines in the CA3 region.
        zscore_CA3 : list
            Z-scores for spines in the CA3 region.
        ts_CA3 : list
            Time series data for spines in the CA3 region.
        dFF_BLA : list
            dF/F traces for spines in the BLA region.
        zscore_BLA : list
            Z-scores for spines in the BLA region.
        ts_BLA : list
            Time series data for spines in the BLA region.
        spines_data : list
            Processed spine data from semantic segmentation.
        calcium_events_BLA : list
            Calcium event probabilities for spines in the BLA region.
        calcium_events_CA3 : list
            Calcium event probabilities for spines in the CA3 region.
        spine_predictions : np.ndarray
            Raw neural network predictions for spine segmentation.
        dendrite_predictions : np.ndarray
            Raw neural network predictions for dendrite segmentation.

        Attributes
        ----------
        roi_index : int
            Index of the ROI.
        roi_metadata : dict
            Metadata associated with the ROI.
        roi : object
            Raw ROI data containing imaging or fluorescence frames.
        dFF_CA3, zscore_CA3, ts_CA3 : list
            dF/F traces, Z-scores, and time series for spines
            in the CA3 region.
        dFF_BLA, zscore_BLA, ts_BLA : list
            dF/F traces, Z-scores, and time series for spines
            in the BLA region.
        spines_data : list
            Processed spine data for the ROI.
        params : dict
            Configuration parameters for segmentation and processing.
        base_image : np.ndarray
            Base image for the ROI, generated from max projection
            of imaging data.
        n_spines : int
            Number of spines segmented within the ROI.

        Notes
        -----
        - Metadata and configuration parameters are added as attributes
            to the instance.
        - The `base_image` used for segmentation is generated
            `get_base_image()`.
        - The number of spines in the ROI is determined and stored
            in `n_spines`.
        """

        self.roi_index = roi_index
        self.roi_metadata = roi_metadata
        self.roi = roi_data

        self.dFF_CA3 = dFF_CA3
        self.zscore_CA3 = zscore_CA3
        self.ts_CA3 = ts_CA3
        self.dFF_BLA = dFF_BLA
        self.zscore_BLA = zscore_BLA
        self.ts_BLA = ts_BLA
        self.spines_data = spines_data
        self.params = params

        self.calcium_events_BLA = calcium_events_BLA
        self.calcium_events_CA3 = calcium_events_CA3

        self.spine_predictions = spine_predictions
        self.dendrite_predictions = dendrite_predictions
        self.dendrites_mask = dendrites_mask
        self.combined_spines_mask = combined_spines_mask

        self.algorithm = algorithm

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        for key, value in params.items():
            setattr(self, key, value)

        self.base_image = (
            denoised_image if denoised_image is not None else self.get_base_image()
        )
        self.processed_image = self.get_base_image()  # Placeholder for future processing steps
        self.denoised_image = denoised_image

    @property
    def n_spines(self) -> int:
        """Number of spines segmented within the ROI."""
        return len(self.spines_data) if self.spines_data is not None else 0

    def get_base_image(self) -> np.ndarray:
        """
        Generate a base image from the ROI for spine segmentation.

        Combines frames across sweeps, subtracts minimum pixel values,
        and generates a max projection image.

        Returns
        -------
        np.ndarray
            The generated base image.
        """

        # Combine all the frames across sweeps
        combined_frames = np.concatenate(
            [self.roi[n_sweep].sweep[:, 1, :, :].astype(np.float32)
             for n_sweep in range(self.roi_metadata['n_sweeps'])],
            axis=0)

        # Subtract the minimum value in each frame
        # TODO: if data are converted to uint 16 previously,
        # the conversion is not required here
        min_values = np.min(
            combined_frames, axis=(1, 2), keepdims=True)
        combined_frames -= min_values

        # Convert to uint16
        combined_frames_uint16 = combined_frames.astype(np.uint16)

        # Generate the max projection image
        base_image = np.max(
            combined_frames_uint16, axis=0)

        return base_image

    @cache
    def _get_spine(self, spine_index: int) -> Spine:
        """
        Retrieve a specific spine's data.

        Parameters
        ----------
        spine_index : int
            Index of the spine.

        Returns
        -------
        object: Spine
            Spine object containing its data and associated time series.
        """

        selected_dFF_CA3 = (
            self.dFF_CA3[spine_index]
            if len(self.dFF_CA3) > 0 else [])

        selected_zscore_CA3 = (
            self.zscore_CA3[spine_index]
            if len(self.zscore_CA3) > 0 else [])

        selected_ts_CA3 = (
            self.ts_CA3[spine_index]
            if len(self.ts_CA3) > 0 else [])

        selected_dFF_BLA = (
            self.dFF_BLA[spine_index]
            if len(self.dFF_BLA) > 0 else [])

        selected_zscore_BLA = (
            self.zscore_BLA[spine_index]
            if len(self.zscore_BLA) > 0 else [])

        selected_ts_BLA = (
            self.ts_BLA[spine_index]
            if len(self.ts_BLA) > 0 else [])

        return Spine(
            spine_index,
            self.spines_data[spine_index],
            self.roi,
            self.roi_metadata,
            selected_dFF_CA3,
            selected_zscore_CA3,
            selected_ts_CA3,
            selected_dFF_BLA,
            selected_zscore_BLA,
            selected_ts_BLA,
            self.params  # Pass params including device info
        )

    def __getitem__(self, spine_index) -> Spine:
        if self.spines_data is None:
            raise KeyError(
                "No spine data available. "
                "Run DatasetSegmenter.collect_all_data() first.")
        return self._get_spine(spine_index)

    def __iter__(self) -> RoiSpine:
        if self.spines_data is None:
            raise KeyError(
                "No spine data available. "
                "Run DatasetSegmenter.collect_all_data() first.")
        self._current_spine_index = 0
        return self

    def __next__(self) -> Spine:
        if self._current_spine_index < len(self.spines_data):
            spine = self[self._current_spine_index]
            self._current_spine_index += 1
            return spine
        else:
            raise StopIteration

    def __len__(self) -> int:
        if self.spines_data is None:
            raise KeyError(
                "No spine data available. "
                "Run DatasetSegmenter.collect_all_data() first.")
        return len(self.spines_data)


class Spine:

    def __init__(
            self,
            spine_index,
            spine_metadata,
            roi_data,
            roi_metadata,
            selected_dFF_CA3,
            selected_zscore_CA3,
            selected_ts_CA3,
            selected_dFF_BLA,
            selected_zscore_BLA,
            selected_ts_BLA,
            params,
    ) -> None:

        self.spine_index = spine_index
        self.metadata = spine_metadata
        self.roi = roi_data
        self.roi_metadata = roi_metadata

        self.dFF_CA3 = selected_dFF_CA3
        self.zscore_CA3 = selected_zscore_CA3
        self.ts_CA3 = selected_ts_CA3
        self.dFF_BLA = selected_dFF_BLA
        self.zscore_BLA = selected_zscore_BLA
        self.ts_BLA = selected_ts_BLA

        for attr, key in self.metadata.items():
            setattr(self, attr, key)

        for attr, key in roi_metadata.items():
            setattr(self, attr, key)

        for attr, key in params.items():
            setattr(self, attr, key)

    def f(
            self,
            sweep_index: int,
            rolling_bsl: str = None,
            window_sec: float = None,
            min_quantile: int = None,
    ) -> np.ndarray:
        """
        Calculate the dF/F (delta F over F) for a specific spine.
        
        Parameters
        ----------
        sweep_index : int
            Index of the sweep to calculate dF/F for.
        rolling_bsl : str, optional
            Rolling baseline method ('centered', 'left', 'right').
            If None, uses value from config.
        window_sec : float, optional
            Window size in seconds for rolling baseline calculation.
            If None, uses value from config.
        min_quantile : int, optional
            Minimum quantile for baseline calculation.
            If None, uses value from config.
        
        Returns
        -------
        np.ndarray
            dF/F trace for the specified sweep.
        """
        # Use config defaults if not provided
        rolling_bsl = rolling_bsl if rolling_bsl is not None else self.rolling_bsl
        window_sec = window_sec if window_sec is not None else self.window_sec
        min_quantile = min_quantile if min_quantile is not None else self.min_quantile
        
        dff_tensor = dFF(
            n_frames=self.n_frames,
            roi=self.roi,
            mask=self.mask,
            frame_rate=self.frame_rate,
            sweep_index=sweep_index,
            rolling_bsl=rolling_bsl,
            window_sec=window_sec,
            min_quantile=min_quantile,
        )
        return dff_tensor.numpy()

    def ft(self) -> np.ndarray:
        """
        Get the time stamps for the spine.
        """
        timestamps_tensor = get_timestamps(
            n_frames=self.n_frames,
            frame_rate=self.frame_rate
        )
        return timestamps_tensor.numpy()

    def zscore(
            self,
            sweep_index: int,
    ) -> np.ndarray:
        """
        Calculate the z-score for a specific spine.
        """
        z_tensor = z_score(
            n_frames=self.n_frames,
            roi=self.roi,
            mask=self.mask,
            frame_rate=self.frame_rate,
            sweep_index=sweep_index
        )
        return z_tensor.numpy()

