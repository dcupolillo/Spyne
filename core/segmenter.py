""" Created on Fri Mar  1 16:20:12 2024
    @author: dcupolillo """

from pathlib import Path
import numpy as np
import tensorflow as tf
import flammkuchen as fl
from tqdm import tqdm
from itertools import chain
from functools import cache
import matplotlib.pyplot as plt
import torch
import random

from spyne.core.time_series.calculate_timeseries import (
    dFF, get_time_series, z_score)
from spyne.neuralnetwork.spine_segmentation.spine_segmentation import (
    inference, process_predictions, pad_images)
from spyne.display.events import (
    plot_spine_pixel_annotation, plot_spine_calcium_traces,
    plot_spine_zscores, plot_single_spine_dFF)
from spyne.display.plot_segmenter import (
    plot_all_spines, plot_events_spines, plot_all_zscore_heatmap,
    spine_sholl)
from spyne.core.utils.detect_event import (
    is_calcium_event, binarize_calcium_events_array)
from spyne.neuralnetwork.zscore_decoder.network import ZScoreNN
from spyne.neuralnetwork.zscore_decoder.utils import set_device


class DatasetSegmenter:

    def __init__(
            self,
            dataset: object,
            model_fn: str or Path = (
                "C:\\Users\\dcupolillo\\Projects\\spyne\\"
                "models\\model_240909_2.h5"),
            spine_threshold: float = 0.3,
            dendrite_threshold: float = 0.7,
            mask_size: int = 3,
            min_distance: int = 5,
            min_spine_size: float = 4,
            min_dendrite_size: float = 15,
            dendrite_dilation_iterations: int = 12,
            kernel_size: int = 3,
            sd_factor: int = 2,
    ) -> None:

        if type(dataset).__name__ != 'ImagingDataset':
            raise TypeError('Invalid input type for dataset')

        if not Path(model_fn).exists():
            raise FileNotFoundError(f'{model_fn} does not exist')

        self.dataset = dataset
        self.metadata = self.dataset.metadata

        self.device = (
            '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')

        self.params = {
            'device': self.device,
            'model_fn': model_fn,
            'spine_threshold': spine_threshold,
            'min_spine_size': min_spine_size,
            'mask_size': mask_size,
            'min_distance': min_distance,
            'dendrite_threshold': dendrite_threshold,
            'min_dendrite_size': min_dendrite_size,
            'dendrite_dilation_iterations': dendrite_dilation_iterations,
            'kernel_size': kernel_size,
            'sd_factor': sd_factor,
        }

        for key, value in self.params.items():
            setattr(self, key, value)

        self.load_batch_data()

    def load_batch_data(self) -> None:
        """
        Load batch data from h5 files if they exist in the
        dataset's parent folder.
        Otherwise, sets the corresponding attributes to None.
        """
        path = self.dataset.folder.parent

        files = {
            'batch_z_scores_CA3': 'batch_z_scores_CA3.h5',
            'batch_dFF_CA3': 'batch_dFF_CA3.h5',
            'batch_ts_CA3': 'batch_ts_CA3.h5',
            'batch_z_scores_BLA': 'batch_z_scores_BLA.h5',
            'batch_dFF_BLA': 'batch_dFF_BLA.h5',
            'batch_ts_BLA': 'batch_ts_BLA.h5',
            'batch_spines_data': 'batch_spines_data.h5',
            'calcium_events_BLA': 'calcium_event_BLA.h5',
            'calcium_events_CA3': 'calcium_event_CA3.h5'
        }

        existing_files = {attr: filename for attr, filename in files.items()
                          if (path / filename).exists()}
        missing_files = {attr: filename for attr, filename in files.items()
                         if not (path / filename).exists()}

        for attr in missing_files:
            setattr(self, attr, [])

        if existing_files:
            for attr, filename in tqdm(existing_files.items(),
                                       desc="Loading batch data",
                                       total=len(existing_files)):
                file_path = path / filename
                try:
                    setattr(self, attr, fl.load(file_path))
                except FileNotFoundError:
                    setattr(self, attr, [])
        (
            self.dynamic_threshold_BLA,
            self.calcium_events_binary_BLA
        ) = binarize_calcium_events_array(self.calcium_events_BLA)
        (
            self.dynamic_threshold_CA3,
            self.calcium_events_binary_CA3
        ) = binarize_calcium_events_array(self.calcium_events_CA3)

        n_spines_BLA = 0
        for spine in self.calcium_events_binary_BLA:
            if sum(spine) > 0:
                n_spines_BLA += 1
        n_spines_CA3 = 0
        for spine in self.calcium_events_binary_CA3:
            if sum(spine) > 0:
                n_spines_CA3 += 1

        self.n_spines_BLA = n_spines_BLA
        self.n_spines_CA3 = n_spines_CA3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, roi_index: int) -> None:

        if roi_index not in self.dataset.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self.dataset.roi_list)}')

        return self.get_roi(roi_index)

    @cache
    def get_roi(self, roi_index: int) -> object:

        roi_metadata = self.metadata[roi_index]

        spine_indices = [
            n for n, i in enumerate(self.batch_spines_data)
            if i['roi_n'] == roi_index]

        selected_dFF_CA3 = (np.array(self.batch_dFF_CA3)[spine_indices]
                            if self.batch_dFF_CA3 else [])
        selected_zscore_CA3 = (np.array(self.batch_z_scores_CA3)[spine_indices]
                               if self.batch_z_scores_CA3 else [])
        selected_ts_CA3 = (np.array(self.batch_ts_CA3)[spine_indices]
                           if self.batch_ts_CA3 else [])
        selected_dFF_BLA = (np.array(self.batch_dFF_BLA)[spine_indices]
                            if self.batch_dFF_BLA else [])
        selected_zscore_BLA = (np.array(self.batch_z_scores_BLA)[spine_indices]
                               if self.batch_z_scores_BLA else [])
        selected_ts_BLA = (np.array(self.batch_ts_BLA)[spine_indices]
                           if self.batch_ts_BLA else [])
        selected_spines_data = [self.batch_spines_data[i]
                                for i in spine_indices]

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
            selected_spines_data,)

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

    def batch_inference(
        self,
        images: np.ndarray or list(np.ndarray),
        segmenters: np.ndarray or list(np.ndarray),
        original_dimensions: tuple or list(tuple),
    ) -> tuple:
        """
        Run inference on a batch of ROI images.

        Parameters
        ----------
        images : Union[tf.Tensor, List[tf.Tensor]]
            The batch of images or a single image.
        segmenters : Union[np.ndarray, List[np.ndarray]]
            The batch of segmenters or a single segmenter.
        original_dimensions : Union[Tuple[int, int], List[Tuple[int, int]]]
            Original dimensions of the images.

        Returns
        -------
        Tuple[dict, dict]
            Spines and dendrites dictionaries.
        """

        with tqdm(
                total=2,
                desc="Inference Progress",
                leave=True) as pbar:

            # Generate batch of predictions
            spine_predictions, dendrite_predictions = inference(
                images,
                self.model_fn,
                original_dimensions=original_dimensions
            )
            pbar.update(1)

            # Process predictions
            predictions = process_predictions(
                segmenters=segmenters,
                spine_predictions=spine_predictions,
                dendrite_predictions=dendrite_predictions,
                spine_threshold=self.spine_threshold,
                dendrite_threshold=self.dendrite_threshold,
                mask_size=self.mask_size,
                min_distance=self.min_distance,
                min_dendrite_size=self.min_dendrite_size,
                dendrite_dilation_iterations=self.dendrite_dilation_iterations,
                min_spine_size=self.min_spine_size,
                kernel_size=self.kernel_size,
                sd_factor=self.sd_factor
            )
            pbar.update(1)

        return predictions

    def collect_all_data(self) -> None:

        images = [None] * len(self.dataset)
        segmenters = [None] * len(self.dataset)

        # Step 1: Collect images and initialize RoiSegmenter instances
        with tf.device(self.device):

            for roi_index in tqdm(
                    range(len(self.dataset)),
                    desc="Collecting images",
                    total=len(self.dataset)):

                segmenter = self[roi_index]  # RoiSegmenter

                segmenters[roi_index] = segmenter
                images[roi_index] = segmenter.get_base_image()

        # Step 2: Run batch inference
        # pad images to get equal size
        padded_images, original_dimensions = pad_images(images)
        padded_images = np.stack(padded_images, axis=0)

        batch_spines_data, batch_dendrites_data = self.batch_inference(
            padded_images, segmenters, original_dimensions)

        # Flatten spines list
        self.batch_spines_data = list(
            chain.from_iterable(batch_spines_data))
        self.batch_dendrites_data = list(
            chain.from_iterable(batch_dendrites_data))

        self.n_spines = len(self.batch_spines_data)

        fl.save(Path(self.dataset.folder.parent, 'batch_spines_data.h5'),
                self.batch_spines_data)

        # Step 3: Link the precomputed spines data to the RoiSegmenter
        spine_counter = 0
        for roi_index, segmenter in enumerate(segmenters):
            num_spines_for_roi = len(batch_spines_data[roi_index])

            if num_spines_for_roi == 0:
                segmenter.spines_data = []
                continue

            segmenter.spines_data = self.batch_spines_data[
                spine_counter:spine_counter + num_spines_for_roi]
            spine_counter += num_spines_for_roi

        # Sanity check to ensure spine data consistency
        assert ((
            [len(i) for i in batch_spines_data]) ==
            ([len(roi.spines_data) for roi in segmenters]))

        total_spines = len(self.batch_spines_data)

        # Initialize lists as attributes to store results for all spines
        self.batch_z_scores_CA3 = []
        self.batch_dFF_CA3 = []
        self.batch_ts_CA3 = []
        self.batch_z_scores_BLA = []
        self.batch_dFF_BLA = []
        self.batch_ts_BLA = []

        with tf.device(self.device):

            with tqdm(total=total_spines,
                      desc="Collecting timeseries",
                      ) as pbar:

                for roi_segmenter, spine_pred in zip(
                        segmenters,
                        self.batch_spines_data):

                    roi_n = roi_segmenter.n_roi
                    n_spines = len(
                        [z for z in self.batch_spines_data
                         if z['roi_n'] == roi_n])

                    if n_spines == 0:
                        continue

                    # Count the number of 'IN 3' and 'IN 5' sweeps
                    n_IN3 = roi_segmenter.adc_list.count('IN 3')
                    n_IN2 = roi_segmenter.adc_list.count('IN 2')

                    for spine_index, spine in enumerate(roi_segmenter):

                        z_scores_CA3 = np.empty(n_IN3, dtype=object)
                        dff_values_CA3 = np.empty(n_IN3, dtype=object)
                        timestamps_CA3 = np.empty(n_IN3, dtype=object)
                        z_scores_BLA = np.empty(n_IN2, dtype=object)
                        dff_values_BLA = np.empty(n_IN2, dtype=object)
                        timestamps_BLA = np.empty(n_IN2, dtype=object)

                        index_CA3 = 0
                        index_BLA = 0

                        for sweep_index in range(roi_segmenter.n_sweeps):

                            adc_value = (roi_segmenter.adc_list[sweep_index])

                            z_score = spine.zscore(sweep_index)
                            dff_value = spine.f(sweep_index)
                            timestamp = spine.ft(sweep_index)

                            if adc_value == 'IN 3':
                                z_scores_CA3[index_CA3] = z_score
                                dff_values_CA3[index_CA3] = dff_value
                                timestamps_CA3[index_CA3] = timestamp
                                index_CA3 += 1

                            elif adc_value == 'IN 2':
                                z_scores_BLA[index_BLA] = z_score
                                dff_values_BLA[index_BLA] = dff_value
                                timestamps_BLA[index_BLA] = timestamp
                                index_BLA += 1
                            else:
                                print("unknown adc")

                        # Sanity check to ensure sweep number consistency
                        assert (index_CA3 + index_BLA) == (sweep_index + 1)

                        # Append the results for the current spine
                        self.batch_z_scores_CA3.append(list(z_scores_CA3))
                        self.batch_dFF_CA3.append(list(dff_values_CA3))
                        self.batch_ts_CA3.append(list(timestamps_CA3))
                        self.batch_z_scores_BLA.append(list(z_scores_BLA))
                        self.batch_dFF_BLA.append(list(dff_values_BLA))
                        self.batch_ts_BLA.append(list(timestamps_BLA))

                        pbar.update(1)

                    if not n_spines == 0:
                        assert spine_index == (n_spines - 1)

        # Save the results
        if self.batch_z_scores_CA3:
            fl.save(
                Path(self.dataset.folder.parent, 'batch_z_scores_CA3.h5'),
                self.batch_z_scores_CA3)
        if self.batch_dFF_CA3:
            fl.save(
                Path(self.dataset.folder.parent, 'batch_dFF_CA3.h5'),
                self.batch_dFF_CA3)
        if self.batch_ts_CA3:
            fl.save(
                Path(self.dataset.folder.parent, 'batch_ts_CA3.h5'),
                self.batch_ts_CA3)
        if self.batch_z_scores_BLA:
            fl.save(
                Path(self.dataset.folder.parent, 'batch_z_scores_BLA.h5'),
                self.batch_z_scores_BLA)
        if self.batch_dFF_BLA:
            fl.save(
                Path(self.dataset.folder.parent, 'batch_dFF_BLA.h5'),
                self.batch_dFF_BLA)
        if self.batch_ts_BLA:
            fl.save(
                Path(self.dataset.folder.parent, 'batch_ts_BLA.h5'),
                self.batch_ts_BLA)

        self.calcium_event_flag()

    def calcium_event_flag(
            self,
            model_path=(
                r"C:/Users/dcupolillo/Projects/spyne/"
                r"neuralnetwork/zscore_decoder/zscore_best_model.pth"),
    ) -> None:

        self.calcium_events_BLA = []
        self.calcium_events_CA3 = []

        model = ZScoreNN()
        model.load_state_dict(torch.load(model_path))
        device = set_device()
        model.to(device)

        # Process BLA
        for spine in tqdm(
                self.batch_z_scores_BLA,
                desc="Analyzing BLA spines"):
            spine_flags_BLA = []
            for sweep in spine:
                spine_flags_BLA.append(
                    is_calcium_event(sweep, model=model, device=device))
            self.calcium_events_BLA.append(spine_flags_BLA)

        # Process CA3
        for spine in tqdm(
                self.batch_z_scores_CA3,
                desc="Analyzing CA3 spines"):
            spine_flags_CA3 = []
            for sweep in spine:
                spine_flags_CA3.append(
                    is_calcium_event(sweep, model=model, device=device))
            self.calcium_events_CA3.append(spine_flags_CA3)

        # Sanity check
        assert len(self.calcium_events_BLA) == len(self.batch_z_scores_BLA)
        for spine_flags, spine_traces in zip(
                self.calcium_events_BLA, self.batch_z_scores_BLA):
            assert len(spine_flags) == len(spine_traces)

        if self.calcium_events_BLA:
            fl.save(Path(self.dataset.folder.parent, 'calcium_event_BLA.h5'),
                    self.calcium_events_BLA)

        if self.calcium_events_CA3:
            fl.save(Path(self.dataset.folder.parent, 'calcium_event_CA3.h5'),
                    self.calcium_events_CA3)

    def plot_all_spines(
            self,
            spine_size: int = 20,
            spine_color: str or tuple = (1, 0, 1),
            ax: plt.Axes = None,
            fontsize: int = 10,
    ) -> None:

        return plot_all_spines(
            all_spines=self.batch_spines_data,
            spine_size=spine_size,
            spine_color=spine_color,
            ax=ax,
            fontsize=fontsize)

    def plot_events_spines(
            self,
            input_type: str = 'BLA',
            n_event_threshold: int = 0,
            spine_size: int = 20,
            ax=None,
            fontsize: int = 10,
            cmap: str = 'viridis',
            show_cmap: bool = True,
            dynamic_threshold_percentile: int = 99,
    ) -> None:

        if input_type == 'BLA':
            events_probability = self.calcium_events_BLA
        elif input_type == 'CA3':
            events_probability = self.calcium_events_CA3
        else:
            raise ValueError("Incorrect input type")

        return plot_events_spines(
            all_spines=self.batch_spines_data,
            events_probability=events_probability,
            n_event_threshold=n_event_threshold,
            spine_size=spine_size,
            ax=ax,
            fontsize=fontsize,
            cmap=cmap,
            show_cmap=show_cmap,
            dynamic_threshold_percentile=dynamic_threshold_percentile)

    def sholl(
            self,
            morphology: object,
            radius_step: float,
            n_radii: int,
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

        if not input_type:
            event_list = None
        elif input_type == 'BLA':
            event_list = self.calcium_events_BLA
        elif input_type == 'CA3':
            event_list = self.calcium_events_CA3
        else:
            raise ValueError("Incorrect input type")

        return spine_sholl(
            all_spines=self.batch_spines_data,
            morphology=morphology,
            events_flag=event_list,
            n_event_threshold=n_event_threshold,
            radius_step=radius_step,
            n_radii=n_radii,
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
            input_type: str = None,
            average: bool = False,
            sort: bool = False):

        return plot_all_zscore_heatmap(
            all_spines=self,
            input_type=input_type,
            average=average,
            sort=sort)


class RoiSegmenter:

    def __init__(
            self,
            roi_index,
            roi_metadata,
            roi_data,
            params,
            dFF_CA3,
            zscore_CA3,
            ts_CA3,
            dFF_BLA,
            zscore_BLA,
            ts_BLA,
            spines_data
    ) -> None:

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

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        for key, value in params.items():
            setattr(self, key, value)

        self.base_image = self.get_base_image()

        self.n_spines = len(self.spines_data)

    def get_base_image(self) -> np.ndarray:
        """
        Generate a base image from the ROI for spine segmentation.
        Randomly sample half of the overall number of combined frames,
        convert values to uint16, and ensure non-negativity of pixel values.
        Generate max projection image.

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

    def inference(self, model_fn: str or Path = None) -> None:

        if self.spines_data is not None:
            return

        if not model_fn:
            model_fn = self.model_fn

        padded_image, original_dimensions = pad_images(self.base_image)

        spine_predictions, dendrite_predictions = inference(
            padded_image,
            model_fn,
            original_dimensions
        )

        self.spines_data, self.dendrites_data = process_predictions(
            segmenters=self.roi,
            spine_predictions=spine_predictions,
            dendrite_predictions=dendrite_predictions,
            spine_threshold=self.spine_threshold,
            dendrite_threshold=self.dendrite_threshold,
            mask_size=self.mask_size,
            min_distance=self.min_distance,
            min_dendrite_size=self.min_dendrite_size,
            dendrite_dilation_iterations=self.dendrite_dilation_iterations,
            min_spine_size=self.min_spine_size,
            kernel_size=self.kernel_size,
            sd_factor=self.sd_factor
        )

        self.n_spines = len(self.spines_data)

    def plot_masks(
            self,
            image_cmap: str = 'binary_r',
            spines_cmap: str = 'gist_rainbow',
            spine_mask_alpha: float = 0.5,
    ):
        if self.spines_data is None:
            raise KeyError("Run inference() first.")

        return plot_spine_pixel_annotation(
            spines=self,
            base_image=self.base_image,
            image_cmap=image_cmap,
            spines_cmap=spines_cmap,
            spine_mask_alpha=spine_mask_alpha)

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
    ):
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
    ):

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

    @cache
    def get_spine(self, spine_index: int) -> object:

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
