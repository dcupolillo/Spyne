""" Created on Tue Oct 24 18:25:12 2023
    @author: dcupolillo """

from pathlib import Path
import numpy as np

from spyne.core.utils.utils import transform
from spyne.core.utils.filters import modified_okada_filter, ewma
from spyne.neuralnetwork.spine_segmentation import (
    spines_and_dendrites_segmentation)
from spyne.display.events import plot_spine_events, plot_single_spine_events


class SpineCollection():

    def __init__(
            self,
            roi: object,
            model_fn: str or Path = 'models\\model_2405.h5',
            spine_threshold: float = 0.1,
            dendrite_threshold: float = 0.7,
            mask_size: int = 3,
            min_distance: int = 3,
            min_spine_size: float = 4,
            min_dendrite_size: float = 15,
            dendrite_dilation_iterations: int = 12,
    ) -> None:
        """
        Representation of a group of spines segmented from a single ROI

        Parameters
        ----------
        roi : spyne.Roi
            DESCRIPTION.
        model_fn: str or Path, optional
            DESCRPTION.
        threshold : float, optional
            DESCRIPTION. The default is 0.02.
        mask_size : int, optional
            DESCRIPTION. The default is 5.
        min_distance : int, optional
            DESCRIPTION. The default is 5.

        Raises
        ------
        KeyError
            For invalid input.

        Returns
        -------
        None
        """

        if type(roi).__name__ != 'Roi':
            raise KeyError('Invalid input')

        if not isinstance(model_fn, Path):
            model_fn = Path(model_fn)

        if not model_fn.exists():
            raise ValueError(f'{model_fn} does not exist')

        self.roi = roi

        self.affine = self.roi.affine
        self.pixeltoreftransform = np.array(self.roi.pixeltoreftransform)
        self.centerxy = self.roi.centerxy
        self.pixelresolutionxy = self.roi.pixelresolutionxy

        self.model_fn = (Path(model_fn)
                         if not isinstance(model_fn, Path)
                         else model_fn)

        self.spine_threshold = spine_threshold
        self.min_spine_size = min_spine_size
        self.mask_size = mask_size
        self.min_distance = min_distance

        self.dendrite_threshold = dendrite_threshold
        self.min_dendrite_size = min_dendrite_size
        self.dendrite_dilation_iterations = dendrite_dilation_iterations

        self.base_image = self.generate_base_image()
        self.get_roi_params()
        self.segment_spines_and_dendrites()

    def generate_base_image(
            self,
    ) -> np.ndarray:
        """
        Generate a base image from the ROI for spine segmentation.
        Combine all frames of all sweeps, convert to uint16
        and ensure non-negativity of pixel values.
        Generate max projection image.

        Parameters
        ----------
        roi : spyne.Roi
            The ROI from which to generate the base image.

        Returns
        -------
        np.ndarray
           The generated base image.

        """

        combined_frames = [self.roi[n_sweep].frame_seq
                           for n_sweep in self.roi.sweep_list]

        N = len(combined_frames)
        frame_shape = combined_frames[0].shape

        combined_frames_uint16 = np.empty((N,) + frame_shape, dtype=np.uint16)

        for i, frame in enumerate(combined_frames):
            # Ensure pixel values are non-negative
            frame -= frame.min()
            combined_frames_uint16[i] = frame.astype(np.uint16)

        combined_sequence = np.concatenate(combined_frames_uint16, axis=0)
        maxproj = np.max(combined_sequence[:, 1, :, :], axis=0)

        return maxproj

    def segment_spines_and_dendrites(
            self,
    ) -> None:
        """
        Semantic segmentation of given image
        to identify spines

        Parameters
        ----------
        base_image : np.ndarray
            The image to apply the model to
        model : str, optional
            DESCRIPTION. The default is 'models\\DeepD3_32F.h5'.
        threshold : float, optional
            DESCRIPTION. The default is 0.02.
        mask_size : int, optional
            DESCRIPTION. The default is 3.
        min_distance : int, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        """

        (
            self.spine_data,
            self.dendrites_data
        ) = spines_and_dendrites_segmentation(
            base_image=self.base_image,
            model_fn=self.model_fn,
            spine_threshold=self.spine_threshold,
            dendrite_threshold=self.dendrite_threshold,
            mask_size=self.mask_size,
            min_distance=self.min_distance,
            min_dendrite_size=self.min_dendrite_size,
            dendrite_dilation_iterations=self.dendrite_dilation_iterations,
            min_spine_size=self.min_spine_size)

        centroids_fov = transform(
            [spine['centroid_pix'] for spine in self.spine_data],
            self.affine,
            self.pixeltoreftransform,
            self.centerxy,
            self.pixelresolutionxy)

        self.spine_data = [{**spine, 'centroid_fov': centroid.tolist()}
                           for spine, centroid in zip(
                                   self.spine_data, centroids_fov)]

        self.spine_list = np.arange(0, len(self.spine_data))
        self.n_spines = len(self.spine_data)

    def get_roi_params(
            self
    ) -> None:
        """
        Gets parameters from the input roi

        Parameters
        ----------
        roi : spyne.Roi
            The input Roi

        Returns
        -------
        None.

        """

        self.pixelresolutionxy = self.roi.pixelresolutionxy
        self.centerxy = self.roi.centerxy

        self.pixeltorreftransform = np.array(self.roi.pixeltoreftransform)
        self.affine = np.array(self.roi.affine)

        self.frame_rate = self.roi.frame_rate
        self.n_sweeps = self.roi.n_sweeps
        self.n_frames = self.roi.n_frames

        self.sweep_list = self.roi.sweep_list

    def plot(
            self,
            cmap: str = 'binary_r',
            sweep_color: str = 'gray',
            spines_cmap: str = 'jet',
     ) -> None:
        return plot_spine_events(
            self,
            cmap=cmap,
            sweep_color=sweep_color,
            spines_cmap=spines_cmap)

    def __getitem__(
            self,
            spine_index: int
    ) -> 'Spine':

        if spine_index not in self.spine_list:
            raise IndexError(f'Spine {spine_index} not in {self.spine_list}')

        return Spine(self.roi, spine_index)


class Spine(SpineCollection):

    def __init__(
            self,
            roi,
            spine_index: int
    ) -> None:
        """
        Represent each individual spine generated with SpineCollection

        Parameters
        ----------
        roi : spyne.Roi
            DESCRIPTION.
        centroids_pix : TYPE
            DESCRIPTION.
        centroids : TYPE
            DESCRIPTION.
        masks : TYPE
            DESCRIPTION.
        spine_index : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        super().__init__(roi)

        self.spine_index = spine_index
        self.data = self.spine_data[spine_index]
        self.mask = self.spine_data[spine_index]['mask']
        self.centroid_pix = self.spine_data[spine_index]['centroid_pix']
        self.centroid_fov = self.spine_data[spine_index]['centroid_fov']

    def f(
            self,
            sweep_index: int,
            rolling_bsl: str = 'centered',
            apply_filter: str = 'okada'
    ) -> np.ndarray:

        f = np.zeros(self.n_frames)

        for n, frame in enumerate(self.roi[sweep_index].frame_seq[:, 0, :, :]):

            frame_min = np.min(frame)

            frame_rescaled = frame - frame_min
            frame_rescaled = frame_rescaled.astype(np.uint16)

            mask_indices = self.mask > 0

            mean_value = np.mean(frame_rescaled[mask_indices])

            f[n] = mean_value

        # Centered rolling quantile
        window_sec = 1
        window = int(window_sec * self.frame_rate)
        half_window = int(window / 2)
        min_quantile = 20

        if rolling_bsl == 'centered':

            baseline = [np.percentile(
                f[t - half_window: t + half_window + 1], min_quantile)
                for t in range(half_window, len(f) - half_window)]

            left_edge_baseline = [np.percentile(
                f[: t + half_window + 1], min_quantile)
                for t in range(half_window)]

            right_edge_baseline = [np.percentile(
                f[t - half_window:], min_quantile)
                for t in range(len(f) - half_window, len(f))]

            bl = np.concatenate(
                (left_edge_baseline, baseline, right_edge_baseline))

        else:

            baseline = [np.percentile(f[t: t + window], min_quantile)
                        for t in range(1, len(f) - window)]

            missing = np.percentile(f[-window:], min_quantile)
            missing = np.repeat(missing, window + 1)

            bl = np.concatenate((baseline, missing))

        dff = ((f - bl) / bl)

        if apply_filter == 'okada':
            dff = modified_okada_filter(dff)

        if apply_filter == 'ewma':
            dff = ewma(dff)

        return dff

    def ft(
            self,
            sweep_index: int
    ) -> np.ndarray:

        return np.array([frame_index / self.frame_rate
                         for frame_index in range(self.n_frames)])

    def plot(
            self,
            mean_color: str = 'green',
            sweep_color: str = 'gray',
            cmap: str = 'binary_r',
            dual_stimulation: bool = False
    ) -> None:

        return plot_single_spine_events(
            self,
            mean_color=mean_color,
            sweep_color=sweep_color,
            cmap=cmap,
            dual_stimulation=dual_stimulation)
