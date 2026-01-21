""" Analysis pipeline for spine processing workflows
    Created on January 21, 2026
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
from spyne.core.imaging.imagingdataset import ImagingDataset
from spyne.core.spines.analysis.segmentation.pipeline import (
    semantic_segmentation_pipeline)
from spyne.core.spines.analysis.timeseries.pipeline import collect_timeseries
from spyne.core.spines.analysis.timeseries.event_detection import detect_calcium_events
from spyne.core.spines.config import SpineDatasetConfig
from spyne.core.spines.dataloader import SpineDataLoader


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
    SEGMENTATION_FILES = {
        'spines_data': 'spines_data.h5',
        'dendrites_data': 'dendrites_data.h5',
        'spine_predictions': 'spine_predictions.h5',
        'dendrite_predictions': 'dendrite_predictions.h5'
    }
    
    TIMESERIES_FILES = {
        'zscores_CA3': 'zscores_CA3.h5',
        'dFF_CA3': 'dFF_CA3.h5',
        'ts_CA3': 'ts_CA3.h5',
        'zscores_BLA': 'zscores_BLA.h5',
        'dFF_BLA': 'dFF_BLA.h5',
        'ts_BLA': 'ts_BLA.h5'
    }
    
    CALCIUM_EVENTS_FILES = {
        'calcium_event_probabilities_BLA': 'calcium_event_probabilities_BLA.h5',
        'calcium_event_probabilities_CA3': 'calcium_event_probabilities_CA3.h5'
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
            'segmentation': False,
            'timeseries': False,
            'calcium_events': False
        }
    
    def run_pipeline(
        self,
        save: bool = True,
        save_path: str or Path = None
    ) -> dict:
        """
        Execute the complete analysis pipeline.
        
        This method runs all processing steps in sequence:
        1. Spine and dendrite segmentation
        2. Time series data collection
        3. Calcium event detection
        
        Parameters
        ----------
        save : bool, optional
            Whether to save intermediate and final results. Default is True.
        save_path : str | Path, optional
            Base directory for saving results. If None, uses dataset folder.
            
        Returns
        -------
        dict
            Dictionary containing all processed data.
        """
        if save_path is None:
            save_path = self.dataset.folder
        
        save_path = Path(save_path)
        
        # Step 1: Spine and dendrite segmentation
        print("Step 1: Running semantic segmentation...")
        segmentation_results = self.run_segmentation_pipeline(
            save=save,
            save_path=save_path / "processed" / "spines"
        )
        
        # Step 2: Time series collection
        print("Step 2: Collecting time series data...")
        timeseries_results = self.run_timeseries_pipeline(
            segmentation_results['spines_data'],
            save=save,
            save_path=save_path / "processed" / "imaging"
        )
        
        # Step 3: Calcium event detection
        print("Step 3: Detecting calcium events...")
        calcium_events_results = self.run_calcium_events_pipeline(
            timeseries_results,
            save=save,
            save_path=save_path / "analysis" / "spines"
        )
        
        # Combine all results
        all_results = {
            **segmentation_results,
            **timeseries_results,
            **calcium_events_results
        }
        
        print("Pipeline completed successfully!")
        return all_results
    
    def run_segmentation_pipeline(
        self,
        save: bool = True,
        save_path: str or Path = None
    ) -> dict:
        """
        Run spine and dendrite segmentation pipeline.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save results. Default is True.
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
        try:
            (
                spine_datasets,
                spines_data,
                dendrites_data,
                spine_predictions,
                dendrite_predictions
            ) = semantic_segmentation_pipeline(
                dataset=self.dataset,
                # FIXME
                spine_dataset=None,  # We'll pass config directly
                config=self.config.segmentation_params
            )
            
            # Store spine datasets for later use
            self._spine_datasets = spine_datasets
            
            # Associate spine data with datasets
            self._associate_spine_data_with_datasets(spine_datasets, spines_data)
            
            results = {
                'spines_data': spines_data,
                'dendrites_data': dendrites_data,
                'spine_predictions': spine_predictions,
                'dendrite_predictions': dendrite_predictions
            }
            
            if save and save_path is not None:
                self._save_segmentation_results(results, save_path)
            
            self._processing_complete['segmentation'] = True
            return results
            
        except Exception as e:
            raise RuntimeError(f"Segmentation pipeline failed: {e}") from e
    
    def run_timeseries_pipeline(
        self,
        spines_data: list,
        save: bool = True,
        save_path: str or Path = None
    ) -> dict:
        """
        Run time series data collection pipeline.
        
        Parameters
        ----------
        spines_data : list
            List of spine data dictionaries from segmentation.
        save : bool, optional
            Whether to save results. Default is True.
        save_path : str or Path, optional
            Directory for saving results.
            
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
        if not self._processing_complete['segmentation']:
            raise ValueError(
                "Segmentation must be completed before time series collection")
        
        if not spines_data:
            raise ValueError("No spine data available for time series collection")
        
        if save and save_path is not None:
            save_path = Path(save_path)
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
                device=self.config.device,
                output_folder=save_path if save else None
            )
            
            results = {
                'zscores_CA3': zscores_CA3,
                'dFF_CA3': dFF_CA3,
                'ts_CA3': ts_CA3,
                'zscores_BLA': zscores_BLA,
                'dFF_BLA': dFF_BLA,
                'ts_BLA': ts_BLA
            }
            
            if save and save_path is not None:
                self._save_timeseries_results(results, save_path)
            
            self._processing_complete['timeseries'] = True
            return results
            
        except Exception as e:
            raise RuntimeError(f"Time series collection failed: {e}") from e
    
    def run_calcium_events_pipeline(
        self,
        timeseries_data: dict,
        save: bool = True,
        save_path: str or Path = None
    ) -> dict:
        """
        Run calcium event detection pipeline.
        
        Parameters
        ----------
        timeseries_data : dict
            Dictionary containing time series data.
        save : bool, optional
            Whether to save results. Default is True.
        save_path : str or Path, optional
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
        if not self._processing_complete['timeseries']:
            raise ValueError(
                "Time series collection must be completed before calcium event detection")
        
        required_keys = ['zscores_BLA', 'dFF_BLA', 'zscores_CA3', 'dFF_CA3']
        missing_keys = [key for key in required_keys if key not in timeseries_data]
        if missing_keys:
            raise ValueError(f"Missing required time series data: {missing_keys}")
        
        if save and save_path is not None:
            save_path = Path(save_path)
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
            
            if save and save_path is not None:
                self._save_calcium_events_results(results, save_path)
            
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
        save_path: str or Path,
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
        save_path: str or Path
    ) -> None:
        """Save segmentation results to files."""
        self._save_results(results, save_path, self.SEGMENTATION_FILES)
    
    def _save_timeseries_results(
        self,
        results: dict,
        save_path: str or Path
    ) -> None:
        """Save time series results to files."""
        self._save_results(results, save_path, self.TIMESERIES_FILES)
    
    def _save_calcium_events_results(
        self,
        results: dict,
        save_path: str or Path
    ) -> None:
        """Save calcium event results to files."""
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
    
    def get_spine_datasets(self) -> Optional[List]:
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