""" Created on Fri Mar  1 16:20:12 2024
    @author: dcupolillo """

from pathlib import Path
import numpy as np
import tensorflow as tf
import flammkuchen as fl
from tqdm import tqdm
from functools import cache
import matplotlib.pyplot as plt

from spyne.display.events import (
    plot_spine_pixel_annotation,
    plot_spine_calcium_traces,
    plot_spine_zscores,
    plot_single_spine_dFF)
from spyne.display.plot_segmenter import (
    plot_spines_2d,
    plot_events_spines,
    plot_zscore_heatmap,
    spine_sholl)
from spyne.core.semantic_segmentation.pipeline import semantic_segmentation_pipeline
from spyne.core.semantic_segmentation.padding import pad_image
from spyne.core.semantic_segmentation.inference import inference
from spyne.core.semantic_segmentation.post_processing import process_predictions
from spyne.core.zscore_classifier.timeseries_pipeline import collect_timeseries
from spyne.core.zscore_classifier.inference import (
    detect_calcium_events,
    binarize_calcium_events_array)
from spyne.core.zscore_classifier.calculate_timeseries import (
    dFF,
    get_time_series,
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
    >>> from neuronpath.path import neuronpath
    >>> paths = neuronpath('YYMMDD', cell_number)
    >>> dataset = spyne.ImagingDataset(paths)
    >>> segmenter = spyne.DatasetSegmenter(dataset)

    >>> # Run segmentation and analyze the dataset
    >>> segmenter.collect_all_data()
    >>> segmenter.detect_calcium_events()

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
                r"models/model_240909_2.h5"),
            spine_threshold: float = 0.3,
            dendrite_threshold: float = 0.7,
            mask_size: int = 3,
            min_distance: int = 5,
            min_spine_size: float = 4,
            min_dendrite_size: float = 15,
            dendrite_dilation_iterations: int = 12,
            kernel_size: int = 3,
            classifier_model_fn: str or Path = (
                r"C:/Users/dcupolillo/Projects/spyne/"
                r"neuralnetwork/zscore_decoder/models/"
                r"zscore_new_best_model.pth"),
            classifier_cutoff: int = 99,
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
            Number of dilation iterations for dendrite segmentation. Default is 12.
        kernel_size : int, optional
            Kernel size for morphological operations. Default is 3.
        classifier_model_fn : str or Path, optional
            Path to the trained model file for calcium event classification.
            Default is "zscore_best_model.pth"
        classifier_cutoff : int, optional
            percentile value to determine calcium probability decision boundary.
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

        if type(dataset).__name__ != 'ImagingDataset':
            raise TypeError('Invalid input type for dataset')

        if not Path(segmentation_model_fn).exists():
            raise FileNotFoundError(f'{segmentation_model_fn} does not exist')

        self.dataset = dataset
        self.metadata = self.dataset.metadata

        # Device for tensorflow-based semantic segmentation
        self.device = (
            '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')

        self.params = {
            'device': self.device,
            'segmentation_model_fn': segmentation_model_fn,
            'spine_threshold': spine_threshold,
            'min_spine_size': min_spine_size,
            'mask_size': mask_size,
            'min_distance': min_distance,
            'dendrite_threshold': dendrite_threshold,
            'min_dendrite_size': min_dendrite_size,
            'dendrite_dilation_iterations': dendrite_dilation_iterations,
            'kernel_size': kernel_size,
            'classifier_model_fn': classifier_model_fn,
            'classifier_cutoff': classifier_cutoff,
        }

        for key, value in self.params.items():
            setattr(self, key, value)

        self.load_data()

    def load_data(self) -> None:
        """
        Load precomputed data (if available) for faster analysis.

        This method checks for the existence of precomputed `.h5` files in the
        dataset's parent directory. If the files exist, they are loaded into memory.
        Otherwise, the corresponding attributes are initialized as empty lists or 
        dynamically generated if sufficient data is available. Newly computed data
        is saved to `.h5` files for future use.

        Attributes Initialized or Updated
        ---------------------------------
        batch_z_scores_CA3 : list
            Z-scores for CA3 spines.
        batch_dFF_CA3 : list
            dF/F values for CA3 spines.
        batch_ts_CA3 : list
            Time series for CA3 spines.
        batch_z_scores_BLA : list
            Z-scores for BLA spines.
        batch_dFF_BLA : list
            dF/F values for BLA spines.
        batch_ts_BLA : list
            Time series for BLA spines.
        calcium_events_BLA : list
            Calcium event probability for BLA spines.
        calcium_events_CA3 : list
            Calcium event probability for CA3 spines.

        Raises
        ------
        FileNotFoundError
            If a specified `.h5` file is not found.
        """
        path = self.dataset.folder.parent

        # TODO: integrate the batch-analysis file name into
        # neuronpath naming system
        files = {
            'zscores_CA3': 'zscores_CA3.h5',
            'dFF_CA3': 'dFF_CA3.h5',
            'ts_CA3': 'ts_CA3.h5',
            'zscores_BLA': 'zscores_BLA.h5',
            'dFF_BLA': 'dFF_BLA.h5',
            'ts_BLA': 'ts_BLA.h5',
            'spines_data': 'spines_data.h5',
            'calcium_events_BLA': 'calcium_events_BLA.h5',
            'calcium_events_CA3': 'calcium_events_CA3.h5',
            'calcium_events_binary_BLA': 'calcium_events_binary_BLA.h5',
            'calcium_events_binary_CA3': 'calcium_events_binary_CA3.h5'
        }

        existing_files = {
            attr: filename for attr, filename in files.items()
            if (path / filename).exists()}
        missing_files = {
            attr: filename for attr, filename in files.items()
            if not (path / filename).exists()}

        for attr in missing_files:
            setattr(self, attr, [])

        if existing_files:
            for attr, filename in tqdm(
                    existing_files.items(),
                    desc="Loading batch data",
                    total=len(existing_files)):

                file_path = path / filename

                try:
                    setattr(self, attr, fl.load(file_path))
                except FileNotFoundError:
                    setattr(self, attr, [])


        self.n_spines_BLA = sum(
            1 for spine in self.calcium_events_binary_BLA
            if sum(spine) > 0)

        self.n_spines_CA3 = sum(
            1 for spine in self.calcium_events_binary_CA3
            if sum(spine) > 0)


    def collect_all_data(self) -> None:
        """
        Collect and process spine and dendrite segmentation
        data for the entire dataset.

        This method performs the following steps:
        1. Runs the semantic segmentation pipeline for all ROIs,
        storing the results in `spines_data` and `dendrites_data`.
        2. Associates segmented spines with their corresponding ROI
        segmenters for indexing.
        3. Collects timeseries data (z-scores, dF/F, timestamps)
        for all segmented spines.
        4. Classifies calcium events based on collected z-scores.
        5. Provides a binary array of activated spines.

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
            dF/F values for BLA spines.
        ts_BLA : np.ndarray
            Timestamps for BLA spines.
        calcium_events_BLA : list
            Calcium event probabilities for BLA spines.
        calcium_events_CA3 : list
            Calcium event probabilities for CA3 spines.
        calcium_events_binary_BLA : list
            Binarized calcium event probabilities for BLA spines.
        calcium_events_binary_CA3 : list
            Binarized calcium event probabilities for CA3 spines.
        dynamic_threshold_BLA : float
            Probability utoff to classify an event for BLA spines.
        dynamic_threshold_CA3 : float
            Probability utoff to classify an event for CA3 spines.

        Notes
        -----
        - Spine and dendrite segmentation is performed using a deep learning model
        configured in `self.params`.
        - Timeseries data collection extracts relevant information (e.g., z-scores, dF/F)
        for all spines detected during segmentation.
        """

        (
            segmenters,
            self.spines_data,
            self.dendrites_data
        ) = semantic_segmentation_pipeline(
            dataset=self.dataset,
            segmenter=self,
            config=self.params,
            output_folder=self.dataset.folder.parent
        )

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

        # Sanity check to ensure spine data consistency
        assert (
            ([len(i) for i in self.spines_data]) ==
            ([len(roi.spines_data) for roi in segmenters]))

        (
            self.zscores_CA3,
            self.dFF_CA3,
            self.ts_CA3,
            self.zscores_BLA,
            self.dFF_BLA,
            self.ts_BLA
        ) = collect_timeseries(
            segmenters,
            self.spines_data,
            self.device,
            self.dataset.folder.parent)

        (
            self.calcium_events_BLA,
            self.calcium_events_CA3
        ) = detect_calcium_events(
            self.params,
            self.zscores_BLA,
            self.zscores_CA3,
            self.dataset.folder.parent
        )

        (
            self.dynamic_threshold_BLA,
            self.calcium_events_binary_BLA
        ) = binarize_calcium_events_array(
            self.calcium_events_BLA,
            self.params['classifier_cutoff'],
        )
        (
            self.dynamic_threshold_CA3,
            self.calcium_events_binary_CA3
        ) = binarize_calcium_events_array(
            self.calcium_events_CA3,
            self.params['classifier_cutoff'],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, roi_index: int) -> None:
        if roi_index not in self.dataset.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self.dataset.roi_list)}')
        return self.get_roi(roi_index)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self.dataset):
            roi_segmenter = RoiSegmenter(self.dataset, self._current_index)
            self._current_index += 1
            return roi_segmenter
        else:
            raise StopIteration

    def spines_by_branch(self, branch_id: int) -> list:
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

        return [
            spine for spine in self.spines_data
            if spine['branch_id'] == branch_id]
    
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
            The data type to use for event analysis ('BLA' or 'CA3'). Default is 'BLA'.
        n_event_threshold : int, optional
            Minimum number of events required for a spine to be considered active.
            Default is 1.
        
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

    def fetch_spine_data(
            self,
            data_batch: list or np.ndarray,
            spine_indices: list
    ) -> np.ndarray:
        """
        Helper function to retrieve data from a given ROI based on spine indices.

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

        return np.array(data_batch)[spine_indices] if data_batch else []

    @ cache
    def get_roi(self, roi_index: int) -> object:
        """
        Retrieve a specific ROI's segmentation and associated data.

        Parameters
        ----------
        roi_index : int
            Index of the ROI to retrieve.

        Returns
        -------
        RoiSegmenter
            An instance containing individual ROI data, metadata, and segmentation.

        Raises
        ------
        IndexError
            If the specified `roi_index` is out of bounds.
        """

        roi_metadata = self.metadata[roi_index]

        spine_indices = [
            n for n, i in enumerate(self.spines_data)
            if i['roi_n'] == roi_index]

        selected_dFF_CA3 = self.fetch_spine_data(
            self.dFF_CA3, spine_indices)
        selected_zscore_CA3 = self.fetch_spine_data(
            self.zscores_CA3, spine_indices)
        selected_ts_CA3 = self.fetch_spine_data(
            self.ts_CA3, spine_indices)
        selected_dFF_BLA = self.fetch_spine_data(
            self.dFF_BLA, spine_indices)
        selected_zscore_BLA = self.fetch_spine_data(
            self.zscores_BLA, spine_indices)
        selected_ts_BLA = self.fetch_spine_data(
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

        return RoiSegmenter(
            roi_index,
            roi_metadata,
            self.dataset[roi_index],
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
            selected_calcium_events_binary_CA3)

    def plot_spines(
            self,
            spines: list = None,
            spine_size: int = 20,
            spine_color: str or tuple = "fuchsia",
            ax: plt.Axes = None,
            fontsize: int = 14,
    ) -> None:
        """
        Visualize detected spines of the dataset
        as scattered dots in the 2D space.

        Parameters
        ----------
        spines : list, optional
            The list of spines to plot. Default is `self.spines_data`.
        spine_size : int, optional
            Size of the spine markers in the plot. Default is 20.
        spine_color : str | tuple, optional
            Color of the spines in the plot. Default is (1, 0, 1) (magenta).
        ax : plt.Axes, optional
            Matplotlib Axes object to plot on. If None, a new figure is created.
            Default is None.
        fontsize : int, optional
            Font size for annotations. Default is 10.

        Returns
        -------
        None
            Displays the plot.
        """

        spines = self.spines_data if not spines else spines

        return plot_spines_2d(
            spines=spines,
            spine_size=spine_size,
            spine_color=spine_color,
            ax=ax,
            fontsize=fontsize)

    def plot_events_spines(
            self,
            spines: list = None,
            input_type: str = 'BLA',
            n_event_threshold: int = 0,
            spine_size: int = 20,
            ax=None,
            fontsize: int = 14,
            cmap: str = 'viridis',
            show_cmap: bool = True,
    ) -> None:
        """
        Plot putative "active" spines with event-based coloring.

        Parameters
        ----------
        spines : list, optional
            List of spines to consider for plotting.
            Default is all spines with `self.spines_data`.
        input_type : str, optional
            The data type to use for event analysis ('BLA' or 'CA3'). Default is 'BLA'.
        n_event_threshold : int, optional
            Minimum number of events required for a spine to be considered active.
            Default is 0.
        spine_size : int, optional
            Size of the markers representing spines in the plot. Default is 20.
        ax : plt.Axes, optional
            Matplotlib Axes object to draw the plot on. If None, a new figure and Axes
            are created. Default is None.
        fontsize : int, optional
            Font size for plot labels and annotations. Default is 10.
        cmap : str, optional
            Colormap used to represent the event counts for each spine. Default is 'viridis'.
        show_cmap : bool, optional
            Whether to display the colormap bar alongside the plot. Default is True.

        Returns
        -------
        None
            The function creates and displays the plot.

        Raises
        ------
        ValueError
            If an invalid `input_type` is provided (must be 'BLA' or 'CA3').

        Notes
        -----
        - This function highlights spines based on their activity levels (calcium events).
        - Active spines are colored according to the number of events they have,
        with the color intensity determined by the `cmap`.
        """

        spines = self.spines_data if not spines else spines

        if input_type == 'BLA':
            events = self.calcium_events_binary_BLA
        elif input_type == 'CA3':
            events = self.calcium_events_binary_CA3
        else:
            raise ValueError("Incorrect input type")

        return plot_events_spines(
            spines=spines,
            events=events,
            n_event_threshold=n_event_threshold,
            spine_size=spine_size,
            ax=ax,
            fontsize=fontsize,
            cmap=cmap,
            show_cmap=show_cmap)

    def sholl(
            self,
            morphology: object,
            radius_step: float,
            n_radii: int,
            spines: list = None,
            input_type: str = None,
            n_event_threshold: int = 0,
            ax: plt.Axes = None,
            ax_sholl_curve: plt.Axes = None,
            circle_color: str = 'gray',
            circle_linestyle: str = 'dashed',
            circle_linewidth: int or float = 1,
            countline_color: str = 'black',
            countline_width: int or float = 1,
            countline_style: str = None,
            countline_alpha: float = 0.5,
            size: int or float = 60,
            fontsize: int = 12,
            cmap: str = 'viridis',
            colorbar_orientation: str = 'vertical'
    ) -> np.ndarray:
        """
        Perform a Sholl analysis of spines.

        Parameters
        ----------
        morphology : object
            Morphology object containing spatial data for spines and dendrites.
        radius_step : float
            Distance between consecutive concentric spheres in the Sholl analysis.
        n_radii : int
            Number of radii (spheres) to generate for the analysis.
        input_type : str, optional
            Specify the data type ('BLA' or 'CA3') for event-based analysis.
            If None, all spines are considered. Default is None.
        n_event_threshold : int, optional
            Minimum number of events required for a spine to be included in the analysis.
            Default is 0.
        ax : plt.Axes, optional
            Matplotlib Axes object for plotting the Sholl analysis in 2D space. Default is None.
        ax_sholl_curve : plt.Axes, optional
            Matplotlib Axes object for plotting the Sholl intersection curve. Default is None.
        circle_color : str, optional
            Color of the concentric circles in the plot. Default is 'gray'.
        circle_linestyle : str, optional
            Linestyle for the concentric circles. Default is 'dashed'.
        circle_linewidth : int or float, optional
            Line width for the concentric circles. Default is 1.
        countline_color : str, optional
            Color of the lines representing intersection counts in the plot. Default is 'black'.
        countline_width : int or float, optional
            Line width for the intersection count lines. Default is 1.
        countline_style : str, optional
            Linestyle for the intersection count lines. Default is None (solid line).
        countline_alpha : float, optional
            Transparency of the intersection count lines. Default is 0.5.
        size : int or float, optional
            Size of the markers representing spines. Default is 60.
        fontsize : int, optional
            Font size for plot annotations and labels. Default is 12.
        cmap : str, optional
            Colormap used to represent spine activity or other properties. Default is 'viridis'.
        colorbar_orientation : str, optional
            Orientation of the colorbar ('vertical' or 'horizontal'). Default is 'vertical'.

        Returns
        -------
        np.ndarray
            Array of intersection counts at each radius.

        Raises
        ------
        ValueError
            If an invalid `input_type` is provided (must be 'BLA', 'CA3', or None).

        Notes
        -----
        - Sholl analysis is used to quantify the distribution of spines or activated spines
        (BLA or CA3) as a function of radial distance from the soma.
        - The concentric circles represent different distances, and the intersections
        quantify how many spines lie within each radius.
        """

        # TODO: having implemented the choice of passing a subset of spines,
        # we should update also the passing of events
        # to reflect the same choice

        spines = self.spines_data if not spines else spines

        if not input_type:
            events = None
        elif input_type == 'BLA':
            events = self.calcium_events_binary_BLA
        elif input_type == 'CA3':
            events = self.calcium_events_binary_CA3
        else:
            raise ValueError("Incorrect input type")

        return spine_sholl(
            morphology=morphology,
            radius_step=radius_step,
            n_radii=n_radii,
            spines=spines,
            events=events,
            n_event_threshold=n_event_threshold,
            ax=ax,
            ax_sholl_curve=ax_sholl_curve,
            circle_color=circle_color,
            circle_linestyle=circle_linestyle,
            circle_linewidth=circle_linewidth,
            countline_color=countline_color,
            countline_width=countline_width,
            countline_style=countline_style,
            countline_alpha=countline_alpha,
            size=size,
            fontsize=fontsize,
            cmap=cmap,
            colorbar_orientation=colorbar_orientation)

    def plot_zscores(
            self,
            spines: list = None,
            input_type: str = None,
            average: bool = False,
            sort: bool = False,
            sort_window: tuple = (16, 21),
            cmap: str = "viridis"
    ) -> None:
        """
        Plot heatmaps of z-scores for spines in the dataset.

        Parameters
        ----------
        spines : list, optional
            List of spines to plot. Default is None, plotting all spines.
        input_type : str, optional
            Specifies the source of z-scores to plot. Expected values are:
            - "CA3" for CA3 spine z-scores.
            - "BLA" for BLA spine z-scores.
            - None (default) to plot z-scores for all input types.
        average : bool, optional
            If True, plot the average z-score across sweeps for each spine 
            (default is False, showing all sweeps).
        sort : bool, optional
            If True, sort spines based on the average z-score before plotting 
            (default is False, maintaining the original order).

        Returns
        -------
        None
            The function generates and displays a heatmap of z-scores using the 
            `plot_all_zscore_heatmap` utility.

        Notes
        -----
        - The heatmap provides a visual representation of z-score distributions 
        for each spine, with color intensity indicating the magnitude of the 
        z-scores.
        - Sorting allows for better visual differentiation of spines based on 
        activity levels.
        - Averaging reduces data dimensionality and highlights overall trends 
        across sweeps.

        Example
        -------
        >>> segmenter = DatasetSegmenter(dataset)
        >>> segmenter.plot_zscores(input_type="CA3", average=True, sort=True)
        """

        # FIXME: the passed spine_data has no intrinsic spine_index.
        # A solution could be adding this to the spine_data
        # or passing the spine_index as a separate list
        
        spines = self.spines_data if not spines else spines

        return plot_zscore_heatmap(
            all_spines_instance=self,
            spines=spines,
            input_type=input_type,
            average=average,
            sort=sort,
            sort_window=sort_window,
            cmap=cmap)


class RoiSegmenter:
    """
    Manage segmentation and analysis of spines and dendrites for a specific ROI.

    This class provides methods for:
    - Performing inference to segment spines and dendrites.
    - Accessing and iterating over segmented spines.
    - Visualizing segmentation results, dF/F traces, and z-scores.

     Example
    -------
    >>> import spyne
    >>> from neuronpath.path import neuronpath
    >>> paths = neuronpath('YYMMDD', cell_number)
    >>> dataset = spyne.ImagingDataset(paths)
    >>> segmenter = spyne.DatasetSegmenter(dataset)

    >>> # Initialize a RoiSegmenter
    >>> segmented_roi = segmenter[0]
    >>> # Display inference result and analisys
    >>> segmented_roi.inference()
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
            calcium_events_binary_CA3: list
    ) -> None:
        """
    Initialize the RoiSegmenter instance for a specific ROI.

    Parameters
    ----------
    roi_index : int
        Index of the region of interest (ROI) within the dataset.
    roi_metadata : dict
        Metadata for the ROI, including information about spatial properties,
        imaging parameters, and number of sweeps.
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
        dF/F traces, Z-scores, and time series for spines in the CA3 region.
    dFF_BLA, zscore_BLA, ts_BLA : list
        dF/F traces, Z-scores, and time series for spines in the BLA region.
    spines_data : list
        Processed spine data for the ROI.
    params : dict
        Configuration parameters for segmentation and processing.
    base_image : np.ndarray
        Base image for the ROI, generated from max projection of imaging data.
    n_spines : int
        Number of spines segmented within the ROI.

    Notes
    -----
    - Metadata and configuration parameters are added as attributes to the instance.
    - The `base_image` used for segmentation is generated `get_base_image()`.
    - The number of spines in the ROI is determined and stored in `n_spines`.
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

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        for key, value in params.items():
            setattr(self, key, value)

        self.base_image = self.get_base_image()

        self.n_spines = len(self.spines_data)

    def get_base_image(self) -> np.ndarray:
        """
        Generate a base image from the ROI for spine segmentation.

        Combines frames across sweeps, subtracts minimum pixel values, and generates a
        max projection image.

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

    def inference(self, segmentation_model_fn: str or Path = None) -> None:
        """
        Perform segmentation inference for spines and dendrites.

        Parameters
        ----------
        segmentation_model_fn : str or Path, optional
            Path to the segmentation model. If not provided, uses `self.segmentation_model_fn`.

        Raises
        ------
        KeyError
            If segmentation results are already present.
        """

        if self.spines_data is not None:
            return

        if not segmentation_model_fn:
            segmentation_model_fn = self.segmentation_model_fn

        padded_image, original_dimensions = pad_image(self.base_image)

        spine_predictions, dendrite_predictions = inference(
            padded_image,
            segmentation_model_fn,
            original_dimensions
        )

        self.spines_data, self.dendrites_data = process_predictions(
            segmenters=self.roi,
            spine_predictions=spine_predictions,
            dendrite_predictions=dendrite_predictions,
            config=self.params
        )

        self.n_spines = len(self.spines_data)

    def plot_masks(
            self,
            image_cmap: str = 'binary_r',
            spines_cmap: str = 'gist_rainbow',
            spine_mask_alpha: float = 0.5,
            enumerate: bool = True,
            enumerate_fontsize: int = 12,
            fontsize: int = 14,
    ) -> None:
        """
        Visualize segmented spine masks over the base image.

        This method overlays the segmented spine masks onto the base image,
        using specified colormaps for the image and spines. The transparency of
        the masks can also be adjusted.

        Parameters
        ----------
        image_cmap : str, optional
            Colormap to apply to the base image. Default is 'binary_r'.
        spines_cmap : str, optional
            Colormap to use for the spine masks. Default is 'gist_rainbow'.
        spine_mask_alpha : float, optional
            Transparency level for the spine masks. Default is 0.5 (50% transparency).
        enumerate : bool, optional
            Enumerate spines with their indices and centroid. Default is True.
        fontsize : int, optional
            The size of the font. Default is 14.

        Returns
        -------
        None
            Displays the plot with overlaid spine masks.

        Raises
        ------
        KeyError
            If no spine data is available (inference has not been performed).

        Notes
        -----
        - This method internally calls `plot_spine_pixel_annotation` for visualization.
        - Ensure that spine data (`self.spines_data`) is available before calling this method.
        """

        if self.spines_data is None:
            raise KeyError("Run inference() first.")

        return plot_spine_pixel_annotation(
            spines=self,
            base_image=self.base_image,
            image_cmap=image_cmap,
            spines_cmap=spines_cmap,
            spine_mask_alpha=spine_mask_alpha,
            enumerate=enumerate,
            enumerate_fontsize=enumerate_fontsize,
            fontsize=fontsize)

    def plot_dFF(
            self,
            input_type: str = None,
            fontsize: int = 10,
            increment: float = 2.,
            spines_cmap: str = 'gist_rainbow',
            sweep_color: str = 'gray',
            sweep_alpha: float = 0.5,
            sweep_linewidth: float = 1.0,
            trace_linewidth: float = 1.5,
            scalebar_y_unit: float = 0.5,
    ) -> None:
        """
        Plot dF/F traces for spines in the ROI.

        Parameters
        ----------
        input_type : str, optional
            Source of the dF/F data (e.g., 'CA3', 'BLA').
        fontsize : int, optional
            Font size for plot labels.
        increment : float, optional
            Vertical spacing between traces.
        spines_cmap : str, optional
            Colormap for spines.
        kwargs : dict
            Additional parameters for plot customization.

        Raises
        ------
        ValueError
            If no spines are available for plotting.
        """

        if self.n_spines == 0:
            raise ValueError("No spines available to plot.")

        return plot_spine_calcium_traces(
            spines=self,
            input_type=input_type,
            fontsize=fontsize,
            increment=increment,
            spines_cmap=spines_cmap,
            sweep_color=sweep_color,
            sweep_alpha=sweep_alpha,
            sweep_linewidth=sweep_linewidth,
            trace_linewidth=trace_linewidth,
            scalebar_y_unit=scalebar_y_unit)

    def plot_zscore(
            self,
            input_type: str = 'CA3',
            zscore_cmap: str = 'viridis',
            fontsize: int = 10,
    ) -> None:
        """
        Plot z-scores for spines in the ROI.

        Parameters
        ----------
        input_type : str, optional
            Source of the z-score data ('CA3' or 'BLA').
        zscore_cmap : str, optional
            Colormap for z-scores.
        fontsize : int, optional
            Font size for plot labels.
        """

        return plot_spine_zscores(
            spines=self,
            input_type=input_type,
            zscore_cmap=zscore_cmap,
            fontsize=fontsize)

    def __len__(self):
        if self.spines_data is None:
            raise KeyError("Run inference() first.")
        return len(self.spines_data)

    def __getitem__(self, spine_index) -> None:
        if self.spines_data is None:
            raise KeyError("Run inference() first.")
        return self.get_spine(spine_index)

    def __iter__(self):
        if self.spines_data is None:
            raise KeyError("Run inference() first.")
        self._current_spine_index = 0
        return self

    def __next__(self):
        if self._current_spine_index < len(self.spines_data):
            spine = self[self._current_spine_index]
            self._current_spine_index += 1
            return spine
        else:
            raise StopIteration

    @ cache
    def get_spine(self, spine_index: int) -> object:
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

        selected_dFF_CA3 = (self.dFF_CA3[spine_index]
                            if len(self.dFF_CA3) > 0 else [])
        selected_zscore_CA3 = (self.zscore_CA3[spine_index]
                               if len(self.zscore_CA3) > 0 else [])
        selected_ts_CA3 = (self.ts_CA3[spine_index]
                           if len(self.ts_CA3) > 0 else [])
        selected_dFF_BLA = (self.dFF_BLA[spine_index]
                            if len(self.dFF_BLA) > 0 else [])
        selected_zscore_BLA = (self.zscore_BLA[spine_index]
                               if len(self.zscore_BLA) > 0 else [])
        selected_ts_BLA = (self.ts_BLA[spine_index]
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

        with tf.device(self.device):

            dff = dFF(
                spine=self,
                sweep_index=sweep_index,
                rolling_bsl=rolling_bsl,
                window_sec=window_sec,
                min_quantile=min_quantile,)

            return dff.numpy()

    def ft(
            self,
            sweep_index: int,
    ) -> np.ndarray:

        with tf.device(self.device):

            timestamps = get_time_series(
                spine=self,
                sweep_index=sweep_index,)

            return timestamps.numpy()

    def zscore(
            self,
            sweep_index: int,
    ) -> np.ndarray:

        with tf.device(self.device):

            z = z_score(
                spine=self,
                sweep_index=sweep_index)

            return z.numpy()

    def plot_dFF(
            self,
            input_type: str = None,
            fontsize: int = 10,
            increment: float = 1.,
            sweep_color: str = 'gray',
            sweep_alpha: float = 0.5,
            mean_color: str = 'green',
            mean_alpha: float = 1.,
            sweep_linewidth: float = 1.0,
            mean_linewidth: float = 1.5,
            scalebar_y_unit: float = 0.5,
    ) -> None:

        return plot_single_spine_dFF(
            self,
            input_type=input_type,
            fontsize=fontsize,
            increment=increment,
            sweep_color=sweep_color,
            sweep_alpha=sweep_alpha,
            mean_color=mean_color,
            mean_alpha=mean_alpha,
            sweep_linewidth=sweep_linewidth,
            mean_linewidth=mean_linewidth,
            scalebar_y_unit=scalebar_y_unit,
        )
