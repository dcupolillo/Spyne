""" Created on Fri Mar  1 16:20:12 2024
    @author: dcupolillo """

from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
import flammkuchen as fl
from tqdm import tqdm
from functools import cache

from spyne.core.imagingdataset import ImagingDataset
from spyne.display.plot_segmenter import rotate_and_transform_spines
from spyne.core.semantic_segmentation.pipeline import (
    semantic_segmentation_pipeline)
from spyne.core.timeseries.timeseries_pipeline import collect_timeseries
from spyne.core.timeseries.inference import (
    detect_calcium_events, binarize_calcium_event_probabilities)
from spyne.core.timeseries.calculate_timeseries import (
    dFF,
    get_timestamps,
    z_score)


class DatasetSegmenter:
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
    >>> segmenter = spyne.DatasetSegmenter(dataset)

    >>> # Run segmentation and analyze the dataset
    >>> segmenter.collect_all_data()
    >>> segmenter.calcium_events_predictions()

    >>> # Visualize detected spines
    >>> segmenter.plot_all_spines()
    >>> segmenter.plot_events_spines(input_type='BLA')
    >>> segmenter.sholl(morphology=morph, radius_step=1, n_radii=10)
    """

    def __init__(
            self,
            dataset: object,
            segmentation_model_fn: str or Path = (
                r"C:/Users/dcupolillo/Projects/spyne/"
                r"inference_models/deepd3/"
                r"model_250414_trial18.h5"),
            spine_threshold: float = 0.3,
            dendrite_threshold: float = 0.7,
            mask_size: int = 3,
            min_distance: int = 5,
            min_spine_size: float = 4,
            min_dendrite_size: float = 15,
            dendrite_dilation_iterations: int = 12,
            classifier_model_fn: str or Path = (
                r"C:/Users/dcupolillo/Projects/spyne/"
                r"inference_models/zscore_classifier/"
                r"your_model_with_dff.pth"),
    ) -> None:
        """
        Initialize the DatasetSegmenter class.

        Parameters
        ----------
        dataset : ImagingDataset
            The imaging dataset to process.
        segmentation_model_fn : str | Path, optional
            Path to the trained model file for segmentation. Default is
            'model_240909_2.h5'.
        spine_threshold : float, optional
            Threshold for spine segmentation. Default is 0.3.
        dendrite_threshold : float, optional
            Threshold for dendrite segmentation. Default is 0.7.
        mask_size : int, optional
            Size of the morphological mask. Default is 3.
        min_distance : int, optional
            Minimum distance for spine separation. Default is 5.
        min_spine_size : float, optional
            Minimum size for spines. Default is 4.
        min_dendrite_size : float, optional
            Minimum size for dendrites. Default is 15.
        dendrite_dilation_iterations : int, optional
            Number of dilation iterations for dendrite segmentation.
            Default is 12.
        classifier_model_fn : str or Path, optional
            Path to the trained model file for calcium event classification.
            Default is "zscore_best_model.pth"
        classifier_cutoff : int, optional
            Percentile value to determine calcium event
            probability decision boundary.
            Default is 99.

        Raises
        ------
        TypeError
            If the `dataset` is not an instance of `ImagingDataset`.
        FileNotFoundError
            If the `segmentation_model_fn` file does not exist.

        Returns
        -------
        None
        """

        if not isinstance(dataset, ImagingDataset):
            raise TypeError('Invalid input type for dataset')

        if not Path(segmentation_model_fn).exists():
            raise FileNotFoundError(f'{segmentation_model_fn} does not exist')

        if not Path(classifier_model_fn).exists():
            raise FileNotFoundError(f'{classifier_model_fn} does not exist')

        self._dataset = dataset
        self.metadata = self._dataset.metadata

        # Device for tensorflow-based semantic segmentation
        self.device = (
            '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')
        self.segmentation_model_fn = segmentation_model_fn
        self.spine_threshold = spine_threshold
        self.min_spine_size = min_spine_size
        self.mask_size = mask_size
        self.min_distance = min_distance
        self.dendrite_threshold = dendrite_threshold
        self.min_dendrite_size = min_dendrite_size
        self.dendrite_dilation_iterations = dendrite_dilation_iterations
        self.classifier_model_fn = classifier_model_fn

        self._load_data()

    @property
    def _files_mapping(self) -> dict:
        """
        Mapping of attribute names to filenames for data loading.

        Returns
        -------
        dict
            Dictionary mapping attribute names to corresponding .h5 filenames.
        """
        return {
            'spines_data': 'spines_data.h5',
            'spine_predictions': 'spine_predictions.h5',
            'dendrite_predictions': 'dendrite_predictions.h5',
            'zscores_CA3': 'zscores_CA3.h5',
            'dFF_CA3': 'dFF_CA3.h5',
            'ts_CA3': 'ts_CA3.h5',
            'zscores_BLA': 'zscores_BLA.h5',
            'dFF_BLA': 'dFF_BLA.h5',
            'ts_BLA': 'ts_BLA.h5',
            'calcium_events_BLA': 'calcium_events_BLA.h5',
            'calcium_events_CA3': 'calcium_events_CA3.h5',
            'calcium_events_binary_BLA':
                'calcium_events_binary_BLA.h5',
            'calcium_events_binary_CA3':
                'calcium_events_binary_CA3.h5',
        }

    def _load_file(
            self,
            filepath: str or Path,
            set_attribute: bool = True
    ) -> tuple[str, any]:
        """
        Load a single .h5 file and optionally set it as an instance attribute.

        This method serves as a building block for loading individual
        data files. It can infer the attribute name from the filename
        using the files mapping or load data from any .h5 file path.

        Parameters
        ----------
        filepath : str or Path
            Path to the .h5 file to load.
        set_attribute : bool, optional
            Whether to set the loaded data as an instance attribute.
            Default is True.

        Returns
        -------
        tuple[str, any]
            Tuple containing (attribute_name, loaded_data).
            If the filename is not in the mapping, attribute_name will be
            the filename without extension.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file is not a .h5 file.
        """
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist")

        if not filepath.suffix == '.h5':
            raise ValueError(f"File {filepath} must be a .h5 file")

        filename = filepath.name
        attribute_name = None

        for attr, mapped_filename in self._files_mapping.items():
            if filename == mapped_filename:
                attribute_name = attr
                break

        if attribute_name is None:
            attribute_name = filepath.stem

        try:
            data = fl.load(filepath)

            if set_attribute:
                setattr(self, attribute_name, data)

            return attribute_name, data

        except Exception:
            if set_attribute:
                setattr(self, attribute_name, [])
            return attribute_name, []

    def load_file(
            self,
            filepath: str or Path
    ) -> tuple:
        """
        Load a single .h5 file and set it as an instance attribute.

        This is a public method that allows loading individual data files
        by pathname. The attribute name is inferred from the filename.

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
        >>> segmenter.load_file('path/to/spines_data.h5')
        ('spines_data', [...])  # Returns attribute name and loaded data

        >>> segmenter.load_file('path/to/custom_analysis.h5')
        ('custom_analysis', [...])  # Custom files get stem as attribute name
        """
        return self._load_file(filepath, set_attribute=True)

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
            'dendrite_dilation_iterations': self.dendrite_dilation_iterations}

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

    def _load_data(
            self,
            path: str or Path = None
    ) -> None:
        """
        Load precomputed data (if available) for faster analysis.

        This method checks for the existence of precomputed `.h5` files in the
        dataset's parent directory. If the files exist, they are loaded
        into memory using the `_load_file` method. Otherwise, the corresponding
        attributes are initialized as empty lists.

        Parameters
        ----------
        path : str or Path, optional
            Path to the directory containing precomputed data files.
            If None, uses the dataset's parent directory.

        Attributes Initialized or Updated
        ---------------------------------
        spines_data : list
            Processed spine data for all ROIs.
        spine_predictions : list
            Raw neural network predictions for spine segmentation.
        dendrite_predictions : list
            Raw neural network predictions for dendrite segmentation.
        zscores_CA3 : list
            Z-scores for CA3 spines.
        dFF_CA3 : list
            dF/F values for CA3 spines.
        ts_CA3 : list
            Time series for CA3 spines.
        zscores_BLA : list
            Z-scores for BLA spines.
        dFF_BLA : list
            dF/F values for BLA spines.
        ts_BLA : list
            Time series for BLA spines.
        calcium_events_BLA : list
            Calcium event probability for BLA spines.
        calcium_events_CA3 : list
            Calcium event probability for CA3 spines.
        calcium_events_binary_BLA : list
            Binarized calcium events for BLA spines.
        calcium_events_binary_CA3 : list
            Binarized calcium events for CA3 spines.
        """
        if path is None:
            path = self._dataset.folder.parent

        if not isinstance(path, Path):
            path = Path(path)

        files = self._files_mapping

        existing_files = {
            attr: filename for attr, filename in files.items()
            if (path / filename).exists()}
        missing_files = {
            attr: filename for attr, filename in files.items()
            if not (path / filename).exists()}

        # Initialize missing files as empty lists
        for attr in missing_files:
            setattr(self, attr, [])

        # Load existing files using _load_file method
        if existing_files:
            for attr, filename in tqdm(
                    existing_files.items(),
                    desc="Loading .h5 data",
                    total=len(existing_files)):

                file_path = path / filename
                self._load_file(file_path, set_attribute=True)

        # Update spine counts
        self.n_spines = len(getattr(self, 'spines_data', []))

        self.n_spines_BLA = sum(
            1 for spine in getattr(self, 'calcium_events_binary_BLA', [])
            if sum(spine) > 0)

        self.n_spines_CA3 = sum(
            1 for spine in getattr(self, 'calcium_events_binary_CA3', [])
            if sum(spine) > 0)

    def collect_all_data(
            self,
            save: bool = True,
            save_path: str or Path = None) -> None:
        """
        Collect and process spine and dendrite segmentation data
        and collects within-spine time series
        for the entire dataset.
        This method performs the following steps:
        1. Runs the semantic segmentation pipeline for all ROIs,
           storing the results in `spines_data` and `dendrites_data`.
        2. Associates segmented spines with their corresponding ROI
           segmenters for indexing.
        3. Collects timeseries data (z-scores, dF/F, timestamps)
        for all segmented spines.
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
        Notes
        -----
        - Spine and dendrite segmentation is performed using a deep learning
          model configured in `self.params`.
        - Timeseries data collection extracts relevant information
            (e.g., z-scores, dF/F) for all spines detected during segmentation.
            """
        output_folder = (
            self._dataset.folder.parent if save_path is None else save_path)

        segmenters = self._collect_spines_and_dendrites_data(
            save=save,
            save_path=output_folder
        )

        self._collect_timeseries(
            save=save,
            save_path=output_folder
        )

        self._calcium_events_predictions(
            save=save,
            save_path=output_folder
        )

    def _collect_spines_and_dendrites_data(
            self,
            save: bool,
            save_path: str or Path = None,
    ) -> None:
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

        Attributes Updated
        ------------------
        spines_data : list
            Processed spine data for all ROIs.
        dendrites_data : list
            Processed dendrite data for all ROIs.
        zscores_CA3 : np.ndarray
            Z-scores for CA3 spines.
        dFF_CA3 : np.ndarray
            dF/F values for CA3 spines.
        ts_CA3 : np.ndarray
            Timestamps for CA3 spines.
        zscores_BLA : np.ndarray
            Z-scores for BLA spines.
        dFF_BLA : np.ndarray
            dF/F0 values for BLA spines.
        ts_BLA : np.ndarray
            Timestamps for BLA spines.

        Notes
        -----
        - Spine and dendrite segmentation is performed using a deep learning
            model configured in `self.params`.
        - Timeseries data collection extracts relevant information
            (e.g., z-scores, dF/F) for all spines detected during segmentation.
        """

        (
            segmenters,
            self.spines_data,
            self.dendrites_data,
            self.spine_predictions,
            self.dendrite_predictions
        ) = semantic_segmentation_pipeline(
            dataset=self._dataset,
            segmenter=self,
            config=self.segmentation_params
        )

        # Store segmenters as instance attribute for later use
        self._segmenters = segmenters

        self.n_spines = len(self.spines_data)

        # Link the precomputed spines data to the RoiSegmenter
        spine_counter = 0
        for roi_index, segmenter in enumerate(segmenters):
            n_spines_per_roi = len(
                [z for z in self.spines_data
                 if z['roi_n'] == roi_index])

            if n_spines_per_roi == 0:
                segmenter.spines_data = []
                continue

            segmenter.spines_data = self.spines_data[
                spine_counter:spine_counter + n_spines_per_roi]
            spine_counter += n_spines_per_roi

        if save:
            saving_folder = (
                self._dataset.folder.parent
                if save_path is None else save_path)

            data_to_save = {
                "spines_data.h5": getattr(self, 'spines_data', []),
                "dendrites_data.h5": getattr(self, 'dendrites_data', []),
                "spine_predictions.h5":
                    getattr(self, 'spine_predictions', []),
                "dendrite_predictions.h5":
                    getattr(self, 'dendrite_predictions', [])
            }

            for filename, data in data_to_save.items():
                self._save_to_h5(data, saving_folder, filename)

        return segmenters

    def _collect_timeseries(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """
        Collect timeseries data (z-scores, dF/F, timestamps) for all spines.
        This method processes the segmented spines to extract relevant
        information, including z-scores, dF/F values, and timestamps.

        Parameters
        ----------
        save : bool, optional
            Set to True to save data in .h5 format. Default is True.
        save_path : str or Path, optional
            Directory where the .h5 files will be saved.
            If None, uses the dataset's parent folder.

        Raises
        ------
        ValueError
            If spine data is not available (run segmentation first).
        """
        # Check dependencies
        if not hasattr(self, 'spines_data') or not self.spines_data:
            raise ValueError(
                "Spine data required. "
                "Run _collect_spines_and_dendrites_data() first.")

        saving_folder = (
            self._dataset.folder.parent if save_path is None else save_path)

        (
            self.zscores_CA3,
            self.dFF_CA3,
            self.ts_CA3,
            self.zscores_BLA,
            self.dFF_BLA,
            self.ts_BLA
        ) = collect_timeseries(
            dataset=self._dataset,
            spines_data=self.spines_data,
            metadata=self.metadata,
            device=self.device,
            output_folder=saving_folder)

        if save:
            data_to_save = {
                "zscores_CA3.h5": getattr(self, 'zscores_CA3', []),
                "dFF_CA3.h5": getattr(self, 'dFF_CA3', []),
                "ts_CA3.h5": getattr(self, 'ts_CA3', []),
                "zscores_BLA.h5": getattr(self, 'zscores_BLA', []),
                "dFF_BLA.h5": getattr(self, 'dFF_BLA', []),
                "ts_BLA.h5": getattr(self, 'ts_BLA', [])
            }

            for filename, data in data_to_save.items():
                self._save_to_h5(data, saving_folder, filename)

    def _calcium_events_predictions(
            self,
            save: bool,
            saving_folder: str or Path = None
    ) -> None:
        """
        Detect calcium events in spines using a trained neural network
        classifier.

        This method processes the z-scored traces for BLA and CA3 spines to
        detect calcium events using a pre-trained neural network classifier.
        The results are stored as probabilities and binarized using the
        configured threshold parameters.

        Parameters
        ----------
        save : bool, optional
            Whether to save the calcium event data to .h5 files.
            Default is True.
        saving_folder : str or Path, optional
            Directory where the .h5 files will be saved.
            If None, uses the dataset's parent folder.

        Attributes Updated
        ------------------
        calcium_event_probabilities_BLA : list
            Raw calcium event probabilities for BLA spines.
        calcium_event_probabilities_CA3 : list
            Raw calcium event probabilities for CA3 spines.
        calcium_event_binary_BLA : np.ndarray
            Binarized calcium events for BLA spines.
        calcium_event_binary_CA3 : np.ndarray
            Binarized calcium events for CA3 spines.
        n_spines_BLA : int
            Number of active BLA spines (with at least one calcium event).
        n_spines_CA3 : int
            Number of active CA3 spines (with at least one calcium event).

        Notes
        -----
        - This method requires that timeseries data has been collected first
              (i.e., `_collect_timeseries()` should be run before this method).
        - The method uses the zscore_classifier to predict calcium events
              from z-scored traces.
        - Binarization is performed using the
            `binarize_calcium_event_probabilities`
            function with parameters from the configuration.
        - This method is typically called automatically
            by `collect_all_data()`.

        Raises
        ------
        ValueError
            If z-scored data is not available (run `collect_all_data()` first).
        """
        # if not hasattr(self, 'zscores_BLA') or not self.zscores_BLA:
        #     raise ValueError(
        #         "Z-scored data not available. "
        #         "Run collect_all_data() first."
        #     )

        self.calcium_event_probabilities_BLA = detect_calcium_events(
            config=self.classification_params,
            zscores=self.zscores_BLA,
            dFF=self.dFF_BLA)

        self.calcium_event_probabilities_CA3 = detect_calcium_events(
            config=self.classification_params,
            zscores=self.zscores_CA3,
            dFF=self.dFF_CA3)

        self.calcium_events_binary_BLA = binarize_calcium_event_probabilities(
            self.calcium_event_probabilities_BLA)

        self.calcium_events_binary_CA3 = binarize_calcium_event_probabilities(
            self.calcium_event_probabilities_CA3)

        # Update spine counts
        self.n_spines_BLA = sum(
            1 for spine in self.calcium_events_binary_BLA
            if sum(spine) > 0)

        self.n_spines_CA3 = sum(
            1 for spine in self.calcium_events_binary_CA3
            if sum(spine) > 0)

        if save:
            saving_folder = (
                self._dataset.folder.parent
                if saving_folder is None else saving_folder)

            calcium_events_to_save = {
                "calcium_event_probabilities_BLA.h5":
                    getattr(self, 'calcium_event_probabilities_BLA', []),
                "calcium_event_probabilities_CA3.h5":
                    getattr(self, 'calcium_event_probabilities_CA3', []),
                "calcium_events_binary_BLA.h5":
                    getattr(self, 'calcium_events_binary_BLA', []),
                "calcium_events_binary_CA3.h5":
                    getattr(self, 'calcium_events_binary_CA3', [])
            }

            for filename, data in calcium_events_to_save.items():
                self._save_to_h5(data, saving_folder, filename)

    def _save_to_h5(
            self,
            data,
            output_folder: str or Path,
            filename: str
    ) -> None:
        """
        Generic function to save data to an .h5 file.

        Parameters
        ----------
        data : any
            Data to save. Only saves if data exists and is not empty.
        output_folder : str or Path
            Directory where the .h5 file will be saved.
        filename : str
            Name of the .h5 file to save.
        """
        if not isinstance(output_folder, Path):
            output_folder = Path(output_folder)

        if not output_folder.exists():
            raise FileNotFoundError(
                f"Output folder {output_folder} does not exist.")

        if not output_folder.is_dir():
            raise NotADirectoryError(
                f"Output path {output_folder} is not a directory.")

        if not filename.endswith('.h5'):
            raise ValueError(
                f"Filename {filename} must end with '.h5'.")

        if data is not None and len(data) > 0:
            output_folder = Path(output_folder)
            fl.save(output_folder / filename, data)
            print(f"Saved {filename} to {output_folder}")

    @cache
    def _get_roi(self, roi_index: int) -> RoiSegmenter:
        """
        Retrieve a specific ROI's segmentation and associated data.

        Parameters
        ----------
        roi_index : int
            Index of the ROI to retrieve.

        Returns
        -------
        RoiSegmenter
            An instance containing individual ROI data, metadata, and
            segmentation.

        Raises
        ------
        IndexError
            If the specified `roi_index` is out of bounds.
        """

        roi_metadata = self.metadata[roi_index]

        spine_indices = [
            n for n, i in enumerate(self.spines_data)
            if i['roi_n'] == roi_index]

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
            prob for n, spine in enumerate(self.calcium_events_BLA)
            for prob in spine if n in spine_indices]
        selected_calcium_events_CA3 = [
            prob for n, spine in enumerate(self.calcium_events_CA3)
            for prob in spine if n in spine_indices]
        selected_calcium_events_binary_BLA = [
            prob for n, spine in enumerate(self.calcium_events_binary_BLA)
            for prob in spine if n in spine_indices]
        selected_calcium_events_binary_CA3 = [
            prob for n, spine in enumerate(self.calcium_events_binary_CA3)
            for prob in spine if n in spine_indices]

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

        return RoiSegmenter(
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
            selected_calcium_events_binary_BLA,
            selected_calcium_events_binary_CA3,
            spine_predictions,
            dendrite_predictions)

    def __getitem__(self, roi_index: int) -> RoiSegmenter:
        if roi_index not in self._dataset.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self._dataset.roi_list)}')
        return self._get_roi(roi_index)

    def __getattr__(self, name: str):
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name: str, value):
        self.__dict__[f"_{name}"] = value

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self) -> RoiSegmenter:
        if self._current_index < len(self._dataset):
            roi_segmenter = RoiSegmenter(self._dataset, self._current_index)
            self._current_index += 1
            return roi_segmenter
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._dataset)

    def spines_by_branch(
            self,
            branch_id: int,
            direction: str = None
    ) -> list:
        """
        Retrieve spines associated with a specific branch ID.

        Parameters
        ----------
        branch_id : int
            Branch ID to filter spines.
        direction: str
            If spines coordinates should be flattened in the 'vertical'
            or 'horizontal' direction. Default is None.

        Returns
        -------
        list
            List of spines associated with the specified branch ID.
            Sublist of self.spines_data.
        """

        if branch_id not in np.arange(self._dataset.sf.n_branches):
            raise IndexError("Branch id out of range.")

        spines = [
            spine for spine in self.spines_data
            if spine['branch_id'] == branch_id]

        if not spines:
            raise ValueError("Selected branch has no detected spines.")

        if not direction:
            return spines

        if direction not in ["horizontal", "vertical"]:
            raise ValueError("Unrecognized direction."
                             "Should be `horizontal` or `vertical`.")

        translation, angle = (
            self._dataset.morph.neuron.get_branch(branch_id).flat(
                direction, return_params=True))

        return rotate_and_transform_spines(
            spines,
            translation,
            angle,
            self._dataset.morph.objective_resolution)

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

        return [
            spine for spine in self.spines_data
            if spine['branch_degree'] == branch_degree]

    def spines_by_calcium(
            self,
            input_type: str = "BLA",
            n_event_threshold: int = 1,
    ) -> list:
        """
        Retrieve spines with a minimum number of calcium events.

        Parameters
        ----------
        input_type : str, optional
            The data type to use for event analysis ('BLA' or 'CA3').
            Default is 'BLA'.
        n_event_threshold : int, optional
            Minimum number of events required for a spine to be
            considered active. Default is 1.

        Returns
        -------
        list
            List of spines with the specified number of calcium events.
            Sublist of self.spines_data.
        """

        if input_type == 'BLA':
            events = self.calcium_events_binary_BLA
        elif input_type == 'CA3':
            events = self.calcium_events_binary_CA3
        else:
            raise ValueError("Incorrect input type")

        return [
            spine for spine, event in zip(self.spines_data, events)
            if sum(event) >= n_event_threshold]

    def _fetch_spine_data(
            self,
            data_batch: list or np.ndarray,
            spine_indices: list
    ) -> np.ndarray:
        """
        Helper function to retrieve data from a given ROI based on
        spine indices.

        Parameters
        ----------
        data_batch : list or np.ndarray
            Batch of spine-related data.
        spine_indices : list[int]
            List of indices corresponding to the spines of the given ROI.

        Returns
        -------
        list or np.ndarray
            Filtered data corresponding to the given ROI.
        """

        return (
            np.array(data_batch)[spine_indices]
            if data_batch is not None and len(data_batch) > 0
            else [])


class RoiSegmenter:
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
            calcium_events_binary_BLA: list,
            calcium_events_binary_CA3: list,
            spine_predictions: np.ndarray,
            dendrite_predictions: np.ndarray
    ) -> None:
        """
        Initialize the RoiSegmenter instance for a specific ROI.

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
        calcium_events_binary_BLA : list
            Binarized calcium event probabilities for BLA spines.
        calcium_events_binary_CA3 : list
            Binarized calcium event probabilities for CA3 spines.

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
        self.calcium_events_binary_BLA = calcium_events_binary_BLA
        self.calcium_events_binary_CA3 = calcium_events_binary_CA3

        self.spine_predictions = spine_predictions
        self.dendrite_predictions = dendrite_predictions

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        for key, value in params.items():
            setattr(self, key, value)

        self.base_image = self.get_base_image()

        self.n_spines = len(self.spines_data)

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

        with tf.device(self.device):

            # Combine all the frames across sweeps
            combined_frames = tf.concat(
                [tf.convert_to_tensor(
                    self.roi[n_sweep].sweep[:, 1, :, :],
                    dtype=tf.float32)
                 for n_sweep in range(self.roi_metadata['n_sweeps'])],
                axis=0)

            # Subtract the minimum value in each frame
            # TODO: if data are converted to uint 16 previously,
            # the conversion is not required here
            min_values = tf.reduce_min(
                combined_frames, axis=(1, 2), keepdims=True)
            combined_frames -= min_values

            # Convert to uint16
            combined_frames_uint16 = tf.cast(
                combined_frames, tf.uint16)

            # Generate the max projection image
            base_image = tf.reduce_max(
                combined_frames_uint16, axis=0)

        return base_image.numpy()

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
            selected_ts_BLA,)

    def __getitem__(self, spine_index) -> Spine:
        if self.spines_data is None:
            raise KeyError(
                "No spine data available. "
                "Run DatasetSegmenter.collect_all_data() first.")
        return self._get_spine(spine_index)

    def __iter__(self) -> RoiSegmenter:
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

        self.device = (
            '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')

        for attr, key in self.metadata.items():
            setattr(self, attr, key)

        for attr, key in roi_metadata.items():
            setattr(self, attr, key)

    def f(
            self,
            sweep_index: int,
            rolling_bsl: str = 'centered',
            window_sec: float = 0.5,
            min_quantile: int = 10,
    ) -> np.ndarray:
        """
        Calculate the dF/F (delta F over F) for a specific spine.

        Parameters
        ----------
        sweep_index : int
            Index of the sweep to calculate dF/F for.
        rolling_bsl : str, optional
            Type of baseline correction to apply. Options are 'centered',
            'forward', or 'backward'. Default is 'centered'.
        window_sec : float, optional
            Size of the rolling window in seconds for baseline correction.
            Default is 0.5 seconds.
        min_quantile : int, optional
            Minimum quantile to use for baseline correction.
            Default is 10 (10th percentile).

        Returns
        -------
        np.ndarray
            The calculated dF/F values for the specified spine and sweep.
        """
        with tf.device(self.device):

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

        Returns
        -------
        np.ndarray
            The time series data for the spine.
        """

        with tf.device(self.device):

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
        Parameters
        ----------
        sweep_index : int
            Index of the sweep to calculate z-score for.
        Returns
        -------
        np.ndarray
            The calculated z-score values for the specified spine and sweep.
        """

        with tf.device(self.device):

            z_tensor = z_score(
                n_frames=self.n_frames,
                roi=self.roi,
                mask=self.mask,
                frame_rate=self.frame_rate,
                sweep_index=sweep_index
            )

            return z_tensor.numpy()

