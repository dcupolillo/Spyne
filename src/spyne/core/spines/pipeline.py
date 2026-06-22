""" Analysis pipeline for spine processing workflows
    Created on January 21, 2026
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
import numpy as np
from spyne.core.imaging.imagingdataset import ImagingDataset
from spyne.core.spines.analysis.segmentation.pipeline import semantic_segmentation_pipeline
from spyne.core.spines.analysis.denoise.pipeline import denoising_pipeline
from spyne.core.spines.analysis.timeseries.pipeline import collect_timeseries
from spyne.core.spines.analysis.timeseries.event_detection import detect_calcium_events
from spyne.core.spines.analysis.backends.csbdeep_backend import CSBDeepBackend
from spyne.core.spines.config import SpineDatasetConfig
from spyne.core.spines.dataloader import SpineDataLoader, DataPaths


class SpineAnalysisPipeline:
    """
    Orchestrates spine analysis workflows.
    
    This class manages the complete analysis pipeline:
    - Semantic segmentation of spines and dendrites
    - Time series data collection
    - Calcium event detection
    - Data coordination between processing steps
    """
    
    # Filename mappings for different analysis stages
    # Reference DataPaths constants for consistency
    # Use base keys (no algorithm suffix) - algorithm is added dynamically during save
    SEGMENTATION_FILES = {
        'spines_data': str(DataPaths.SPINES_DATA),
        'dendrites_data': str(DataPaths.DENDRITES_DATA),
        'spine_predictions': str(DataPaths.SPINE_PREDICTIONS),
        'dendrite_predictions': str(DataPaths.DENDRITE_PREDICTIONS),
        'dendrites_masks': str(DataPaths.DENDRITE_MASKS),
        'combined_spines_masks': str(DataPaths.SPINE_MASKS)
    }
    
    TIMESERIES_FILES = {
        'zscores_CA3': str(DataPaths.ZSCORES_CA3),
        'dFF_CA3': str(DataPaths.DFF_CA3),
        'ts_CA3': str(DataPaths.TS_CA3),
        'zscores_BLA': str(DataPaths.ZSCORES_BLA),
        'dFF_BLA': str(DataPaths.DFF_BLA),
        'ts_BLA': str(DataPaths.TS_BLA)
    }
    
    CALCIUM_EVENTS_FILES = {
        'calcium_event_probabilities_BLA': str(DataPaths.CALCIUM_EVENTS_BLA),
        'calcium_event_probabilities_CA3': str(DataPaths.CALCIUM_EVENTS_CA3)
    }
    
    def __init__(
        self,
        dataset: ImagingDataset,
        config: SpineDatasetConfig,
        data_loader: SpineDataLoader
    ):
        """
        Initialize analysis pipeline.
        
        Parameters
        ----------
        dataset : ImagingDataset
            The imaging dataset to process.
        config : SpineDatasetConfig
            Configuration manager instance.
        data_loader : SpineDataLoader
            Data loader instance for I/O operations.
        """
        self.dataset = dataset
        self.config = config
        self.data_loader = data_loader
        
        # Pipeline state
        self._spine_datasets = None
        self._processing_complete = {
            'denoising': False,
            'segmentation': False,
            'timeseries': False,
            'calcium_events': False
        }

    def run_pipeline(
        self,
        spine_dataset,
        save: bool = True,
        save_path: str | Path = None,
        segmentation_algorithm: str = None,
    ) -> dict:
        """
        Execute the complete analysis pipeline.

        This method orchestrates the complete workflow by calling each
        sub-pipeline in sequence. Each sub-pipeline is self-contained and
        handles its own data preparation (image collection, padding, etc.).

        Processing steps:
        0. CARE denoising (optional, if enabled)
           - Collects images from ROIs
           - Pads to uniform dimensions
           - Runs denoising, saves unpadded results to .h5
        1. Spine and dendrite segmentation
           - Collects images (or loads denoised if available)
           - Pads for batch processing
           - Runs segmentation
        2. Time series data collection
        3. Calcium event detection

        Default saves to a "processed" subdirectory within the dataset folder,
        organized by analysis stage.
        
        Parameters
        ----------
        spine_dataset : SpineDataset
            The SpineDataset instance to process.
        save : bool, optional
            Whether to save intermediate and final results. Default is True.
        save_path : str | Path, optional
            Base directory for saving results. If None, uses dataset folder.
        segmentation_algorithm : str, optional
            Algorithm to use for segmentation ('deepd3' or 'nnunet').
            
        Returns
        -------
        dict
            Dictionary containing all processed data.
        """
        if save_path is None:
            save_path = self.dataset.folder
        
        save_path = Path(save_path)
        
        # Step 0: Denoising (optional, if enabled in config)
        if self.config.care_enabled:
            print("Step 0: Running CARE denoising...")
            self.run_denoising_pipeline(
                spine_dataset=spine_dataset,
                dataset_name=self.dataset.name,
                save=save,
                save_path=save_path / "processed" / "imaging",
                overwrite=False
            )
        
        # Step 1: Spine and dendrite segmentation
        print("Step 1: Running spine and dendrite segmentation...")
        segmentation_results = self.run_segmentation_pipeline(
            spine_dataset=spine_dataset,
            save=save,
            save_path=save_path / "processed" / "spines",
            algorithm=segmentation_algorithm
        )
        
        # Step 2: Time series collection
        print("Step 2: Collecting time series data...")
        timeseries_results = self.run_timeseries_pipeline(
            segmentation_results['spines_data'],
            dendrites_masks=segmentation_results['dendrites_masks'],
            save=save,
            save_path=save_path / "processed" / "imaging",
            algorithm=segmentation_algorithm
        )
        
        # Step 3: Calcium event detection
        print("Step 3: Detecting calcium events...")
        calcium_events_results = self.run_calcium_events_pipeline(
            timeseries_results,
            save=save,
            save_path=save_path / "analysis" / "spines",
            algorithm=segmentation_algorithm
        )
        
        # Combine all results
        all_results = {
            **segmentation_results,
            **timeseries_results,
            **calcium_events_results
        }
        
        print("Pipeline completed successfully!")
        return all_results
    
    def run_denoising_pipeline(
        self,
        spine_dataset,
        save: bool = True,
        save_path: str | Path = None,
        overwrite: bool = False
    ) -> np.ndarray:
        """
        Run image denoising pipeline using CARE (Content-Aware restoration).

        This pipeline is self-contained and handles all necessary preparation:
        - Collects images from ROIs
        - Pads images to uniform dimensions
        - Runs CARE denoising
        - Unpads and saves results

        Parameters
        ----------
        spine_dataset : SpineDataset
            The SpineDataset instance to collect images from.
        save : bool, optional
            Whether to save denoised images. Default is True.
        save_path : str | Path, optional
            Directory where result files will be saved.
            If None, uses dataset folder paths.
        overwrite : bool, optional
            Whether to overwrite existing denoised images.
            Default is False.

        Returns
        -------
        np.ndarray
            Denoised images with original (unpadded) dimensions.

        Raises
        ------
        ValueError
            If CARE is not enabled in config or images not provided.
        FileExistsError
            If denoised images already exist and overwrite=False.
        RuntimeError
            If denoising pipeline fails.

        Notes
        -----
        - Saves denoised images as h5 file for programmatic access.
        - Output path: processed/imaging/care_denoised_images.h5
        - Temporary folders (padded inputs, CARE outputs) are deleted
          after the h5 is written.
        - Can load saved results using CSBDeepBackend.load_denoised_h5()
        """
        
        # Check if CARE is enabled in config
        if not self.config.care_enabled:
            raise ValueError(
                "CARE denoising is disabled in configuration. "
                "Set 'denoising.care_parameters.enabled: true' in "
                "spine_config.yaml to enable."
            )

        # Get output paths
        base = Path(save_path) if save_path else self.dataset.folder
        care_output_folder = base / 'processed' / 'imaging' / 'care_denoised'
        care_h5_path = (
            care_output_folder.parent / 'care_denoised_images.h5'
            if save_path
            else self.data_loader._files_mapping['care_denoised_images']
        )

        # Check if denoised images already exist
        if not overwrite and care_h5_path.exists():
            print(f"Denoised images already exist at {care_h5_path}")
            print("Loading existing denoised images...")
            return CSBDeepBackend.load_denoised_h5(care_h5_path)

        # Prepare folders
        care_output_folder.mkdir(parents=True, exist_ok=True)
        padded_images_folder = self.data_loader._files_mapping[
            'padded_images_folder']

        # Prepare config dict for denoising pipeline
        denoising_config = {
            'care_model_fn': self.config.care_model_fn,
            'care_axes': self.config.care_axes,
            'care_python_executable': self.config.care_python_executable
        }

        try:
            denoised_images = denoising_pipeline(
                spine_dataset=spine_dataset,
                dataset=self.dataset,
                config=denoising_config,
                input_folder=padded_images_folder,
                output_folder=care_output_folder,
                save_h5=save,
                h5_output_path=care_h5_path if save else None
            )

            if save:
                print(f"Denoised images saved to: {care_h5_path}")

            return denoised_images

        except Exception as e:
            raise RuntimeError(f"Denoising pipeline failed: {e}") from e
    
    def run_segmentation_pipeline(
        self,
        spine_dataset,
        save: bool = True,
        save_path: str | Path = None,
        algorithm: str = None
    ) -> dict:
        """
        Run spine and dendrite segmentation pipeline.

        This pipeline is self-contained and can run standalone. It:
        - Checks if denoised images exist (from prior denoising run)
        - If yes: loads denoised images
        - If no: collects raw images from ROIs
        - Pads images for batch processing
        - Runs segmentation inference

        Parameters
        ----------
        spine_dataset : SpineDataset
            The SpineDataset instance, used to access RoiSpine objects
            during segmentation.
        save_path : str | Path, optional
            Directory for saving results.
            
        Returns
        -------
        dict
            Dictionary containing segmentation results.
            
        Raises
        ------
        RuntimeError
            If segmentation pipeline fails.
        """
        if save_path is None:
            save_path = self.dataset.folder
        save_path = Path(save_path)
        if save:
            save_path.mkdir(parents=True, exist_ok=True)

        algorithm = algorithm if algorithm is not None else self.config.segmentation_params.get(
            'segmentation_algorithm', 'deepd3').lower()

        # `algorithm` may carry a denoising suffix (e.g. 'nnunet_care') used
        # for file naming. Strip it to get the bare segmentation algorithm
        # passed to the inner pipeline, which only knows 'deepd3' / 'nnunet'.
        supported_algos = self.config.supported_algorithms
        base_algorithm = next(
            (a for a in supported_algos if algorithm == a
             or algorithm.startswith(f"{a}_")),
            algorithm
        )

        # Prepare config dict with resolved model paths and runtime overrides
        config_dict = self.config.config.copy()
        config_dict['models']['deepd3_segmentation']['path'] = str(self.config.deepd3_model_fn)
        config_dict['models']['nnunet_segmentation']['path'] = str(self.config.nnunet_model_fn)
        # Propagate runtime care_enabled override into the raw config dict
        config_dict.setdefault('denoising', {}).setdefault(
            'care_parameters', {})['enabled'] = self.config.care_enabled
        
        try:
            (
                spine_datasets,
                spines_data,
                dendrites_data,
                spine_predictions,
                dendrite_predictions,
                dendrites_masks,
                combined_spines_masks
            ) = semantic_segmentation_pipeline(
                dataset=self.dataset,
                spine_dataset=spine_dataset,
                config=config_dict,
                algorithm=base_algorithm,
            )
            
            # Store spine datasets for later use
            self._spine_datasets = spine_datasets
            
            # Associate spine data with datasets
            self._associate_spine_data_with_datasets(spine_datasets, spines_data)

            # TODO: to be implemented for nnU-Net results as well
            results = {
                'spines_data': spines_data,
                'dendrites_data': dendrites_data,
                'spine_predictions': spine_predictions,
                'dendrite_predictions': dendrite_predictions,
                'dendrites_masks': dendrites_masks,
                'combined_spines_masks': combined_spines_masks,
            }
            
            if save:
                self._save_segmentation_results(results, save_path, algorithm)
            
            self._processing_complete['segmentation'] = True
            return results
            
        except Exception as e:
            raise RuntimeError(f"Segmentation pipeline failed: {e}") from e
    
    def run_timeseries_pipeline(
        self,
        spines_data: list,
        dendrites_masks: list,
        save: bool = True,
        save_path: str | Path = None,
        algorithm: str = None
    ) -> dict:
        """
        Run time series data collection pipeline.

        Parameters
        ----------
        spines_data : list
            List of spine data dictionaries from segmentation.
        dendrites_masks : list
            Per-ROI dendrite masks, indexed by roi_n.
        save : bool, optional
            Whether to save results. Default is True.
        save_path : str | Path, optional
            Directory for saving results.
        algorithm : str, optional
            Algorithm name for file naming. If None, uses default naming.
            
        Returns
        -------
        dict
            Dictionary containing time series results.
            
        Raises
        ------
        ValueError
            If segmentation hasn't been run first.
        RuntimeError
            If time series collection fails.
        """
        if not spines_data:
            raise ValueError(
                "No spine data provided. Run segmentation first or load "
                "spines_data from disk before calling this method."
            )

        if save_path is None:
            save_path = self.dataset.folder
        save_path = Path(save_path)
        
        if save:
            save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            (
                zscores_CA3,
                dFF_CA3,
                ts_CA3,
                zscores_BLA,
                dFF_BLA,
                ts_BLA
            ) = collect_timeseries(
                dataset=self.dataset,
                spines_data=spines_data,
                metadata=self.dataset.metadata,
                dendrites_masks=dendrites_masks,
                device=self.config.device,
                background_dilation_disk_size=(
                    self.config.background_dilation_disk_size),
                neuropil_factor=self.config.neuropil_factor,
                baseline_percentile=self.config.baseline_percentile,
            )
            
            results = {
                'zscores_CA3': zscores_CA3,
                'dFF_CA3': dFF_CA3,
                'ts_CA3': ts_CA3,
                'zscores_BLA': zscores_BLA,
                'dFF_BLA': dFF_BLA,
                'ts_BLA': ts_BLA
            }
            
            if save:
                self._save_timeseries_results(results, save_path, algorithm)
            
            self._processing_complete['timeseries'] = True
            return results
            
        except Exception as e:
            raise RuntimeError(f"Time series collection failed: {e}") from e
    
    def run_calcium_events_pipeline(
        self,
        timeseries_data: dict,
        save: bool = True,
        save_path: str | Path = None,
        algorithm: str = None
    ) -> dict:
        """
        Run calcium event detection pipeline.
        
        Parameters
        ----------
        timeseries_data : dict
            Dictionary containing time series data.
        save : bool, optional
            Whether to save results. Default is True.
        save_path : str | Path, optional
            Directory for saving results.
        algorithm : str, optional
            Algorithm name for file naming. If None, uses default naming.
        
        Returns
        ----------
        timeseries_data : dict
            Dictionary containing time series data.
        save : bool, optional
            Whether to save results. Default is True.
        save_path : str | Path, optional
            Directory for saving results.
            
        Returns
        -------
        dict
            Dictionary containing calcium event detection results.
            
        Raises
        ------
        ValueError
            If required time series data is missing.
        RuntimeError
            If calcium event detection fails.
        """
        required_keys = ['zscores_BLA', 'dFF_BLA', 'zscores_CA3', 'dFF_CA3']
        missing_keys = [key for key in required_keys if key not in timeseries_data]
        if missing_keys:
            raise ValueError(f"Missing required time series data: {missing_keys}")
        
        if save_path is None:
            save_path = self.dataset.folder
        save_path = Path(save_path)
        
        if save:
            save_path.mkdir(parents=True, exist_ok=True)

        try:
            calcium_event_probabilities_BLA = detect_calcium_events(
                config=self.config.classification_params,
                zscores=timeseries_data['zscores_BLA'],
                dFF=timeseries_data['dFF_BLA']
            )
            
            calcium_event_probabilities_CA3 = detect_calcium_events(
                config=self.config.classification_params,
                zscores=timeseries_data['zscores_CA3'],
                dFF=timeseries_data['dFF_CA3']
            )
            
            results = {
                'calcium_event_probabilities_BLA': calcium_event_probabilities_BLA,
                'calcium_event_probabilities_CA3': calcium_event_probabilities_CA3
            }
            
            if save:
                self._save_calcium_events_results(results, save_path, algorithm)
            
            self._processing_complete['calcium_events'] = True
            return results
            
        except Exception as e:
            raise RuntimeError(f"Calcium event detection failed: {e}") from e
    
    def _associate_spine_data_with_datasets(
        self,
        spine_datasets: list,
        spines_data: list
    ) -> None:
        """
        Associate precomputed spine data with their corresponding datasets for indexing.
        This is basically to add the 'roi_n' attribute to each spine in the datasets.
        In addition, it slices the spines_data list to assign the correct spines
        
        Parameters
        ----------
        spine_datasets : list
            List of spine dataset objects.
        spines_data : list
            List of spine data dictionaries.
        """
        spine_counter = 0
        
        for roi_index, segmenter in enumerate(spine_datasets):
            # Count spines for this ROI
            n_spines_per_roi = sum(
                1 for spine in spines_data 
                if spine.get('roi_n') == roi_index
            )
            
            if n_spines_per_roi == 0:
                segmenter.spines_data = []
                continue
            
            # Assign spine data slice to this ROI's segmenter
            segmenter.spines_data = spines_data[
                spine_counter:spine_counter + n_spines_per_roi
            ]
            spine_counter += n_spines_per_roi
    
    def _save_results(
        self,
        results: dict,
        save_path: str | Path,
        filename_mapping: dict = None
    ) -> None:
        """
        Generic method to save analysis results to files.
        
        Parameters
        ----------
        results : dict
            Dictionary containing analysis results.
        save_path : str | Path
            Directory where files will be saved.
        filename_mapping : dict, optional
            Mapping from attribute names to filenames.
            If None, uses DataLoader's default mapping.
        """
        self.data_loader.save_multiple_files(
            results, save_path, filename_mapping
        )
    
    def _save_segmentation_results(
        self,
        results: dict,
        save_path: str | Path,
        algorithm: str
    ) -> None:
        """
        Save segmentation results to files with algorithm suffix.
        
        Parameters
        ----------
        results : dict
            Dictionary of results to save.
        save_path : str | Path
            Directory path for saving files.
        algorithm : str
            Algorithm name to append to filenames.

        Example
        -------
        If algorithm is 'deepd3', 'spines_data' will be saved as 'spines_data_deepd3.h5'.
        """

        algorithm_specific_files = {
            key: str(Path(filename).parent / f"{Path(filename).stem}_{algorithm}{Path(filename).suffix}")
            for key, filename in self.SEGMENTATION_FILES.items()
        }
        self._save_results(results, save_path, algorithm_specific_files)
    
    def _save_timeseries_results(
        self,
        results: dict,
        save_path: str | Path,
        algorithm: str = None
    ) -> None:
        """
        Save time series results to files with optional algorithm suffix.
        
        Parameters
        ----------
        results : dict
            Dictionary of results to save.
        save_path : str | Path
            Directory path for saving files.
        algorithm : str, optional
            Algorithm name to append to filenames.
        """
        if algorithm:
            algorithm_specific_files = {
                key: str(Path(filename).parent / f"{Path(filename).stem}_{algorithm}{Path(filename).suffix}")
                for key, filename in self.TIMESERIES_FILES.items()
            }
            self._save_results(results, save_path, algorithm_specific_files)
        else:
            self._save_results(results, save_path, self.TIMESERIES_FILES)
    
    def _save_calcium_events_results(
        self,
        results: dict,
        save_path: str | Path,
        algorithm: str = None
    ) -> None:
        """
        Save calcium event results to files with optional algorithm suffix.
        
        Parameters
        ----------
        results : dict
            Dictionary of results to save.
        save_path : str | Path
            Directory path for saving files.
        algorithm : str, optional
            Algorithm name to append to filenames.
        """
        if algorithm:
            algorithm_specific_files = {
                key: str(Path(filename).parent / f"{Path(filename).stem}_{algorithm}{Path(filename).suffix}")
                for key, filename in self.CALCIUM_EVENTS_FILES.items()
            }
            self._save_results(results, save_path, algorithm_specific_files)
        else:
            self._save_results(results, save_path, self.CALCIUM_EVENTS_FILES)
    
    @property
    def is_segmentation_complete(self) -> bool:
        """Check if segmentation pipeline has been completed."""
        return self._processing_complete['segmentation']
    
    @property
    def is_timeseries_complete(self) -> bool:
        """Check if time series collection has been completed."""
        return self._processing_complete['timeseries']
    
    @property
    def is_calcium_events_complete(self) -> bool:
        """Check if calcium event detection has been completed."""
        return self._processing_complete['calcium_events']
    
    @property
    def is_pipeline_complete(self) -> bool:
        """Check if the complete pipeline has been executed."""
        return all(self._processing_complete.values())
    
    def get_spine_datasets(self) -> list | None:
        """
        Get spine datasets from segmentation pipeline.
        
        Returns
        -------
        List, optional
            List of spine datasets if segmentation is complete, None otherwise.
        """
        if self._processing_complete['segmentation']:
            return self._spine_datasets
        return None
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state to allow re-running."""
        self._spine_datasets = None
        self._processing_complete = {
            'segmentation': False,
            'timeseries': False,
            'calcium_events': False
        }