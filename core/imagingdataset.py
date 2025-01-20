""" Created on Mon Oct 30 13:59:21 2023
    @author: dcupolillo """

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from functools import cache

from skimage.util import img_as_uint
import skimage
import ROIpy as rp
from spyne.display.plot_frames import (
    animate_frames, save_frames, save_single_frame)
from spyne.core.utils.filters import median, gaussian
from spyne.core.utils.load import load_metadata_from_tiff, load_imaging_data_from_tiff
from neuronpath.path import NeuronPath


class ImagingDataset:
    """
    Manager for imaging data from dendritic ROIs acquired with ScanImage.

    This class loads and processes imaging data from a single neuron,
    including its metadata and corresponding acquisition files. It applies a
    3D median filter to the raw data and corrects the gating-PMT black stripe
    artifact caused by their rapid switching.

    Files includes grouped coplanar ROIs, stacked in vertical.
    Cropping to individual ROIs is provided in the Roi class.

    Depends on file organizer neuronpath.

    Example
    -------
    >>> import spyne
    >>> from neuronpath.path import neuronpath
    >>> paths = neuronpath('YYMMDD', cell_number)
    >>> dataset = spyne.ImagingDataset(paths)
    >>> print(f"Number of ROIs: {len(dataset)}")
    """

    def __init__(
            self,
            paths: NeuronPath,
            kernel_size: tuple = (3, 3, 3)
    ) -> None:
        """
        Initialize the ImagingDataset.

        Parameters
        ----------
        paths : NeuronPath
            Path manager containing paths to imaging and associated files.
        kernel_size : tuple, optional
            Size of the 3D median filter applied to the imaging data.
            Default is (3, 3, 3).

        Raises
        ------
        TypeError
            If `paths` is not a NeuronPath object or `kernel_size` is not a tuple.
        Exception
            If the imaging path is invalid, non-existent, or empty.
        """

        assert paths
        self.paths = paths

        if not isinstance(paths, NeuronPath):
            raise TypeError("'paths' must be a neuronpath.path.NeuronPath")

        if not paths.imaging.is_dir():
            raise Exception(f"{paths.imaging} must be a path to a FOLDER.")

        if not paths.imaging.exists():
            raise Exception(f"{paths.imaging} does not exist")

        if not any(paths.imaging.iterdir()):
            raise Exception(f"{paths.imaging} is empty")

        if not isinstance(kernel_size, tuple):
            raise TypeError("'kernel_size' must be a tuple")

        self.paths = paths
        self.folder = paths.imaging
        self.parent_folder = paths.parent
        self.file_list = paths.tif_file_list
        self.abf_file_list = paths.abf_file_list

        self.median_filter_kernel_size = kernel_size

        self.sf = rp.Scanfields(paths)

        self.metadata = self._load_metadata()
        self.data = self._load_data()

    def _load_metadata(self):
        """
        Load metadata for all ROIs.

        This method extracts information from acquisition files.
        Uses load_metadata_from_tiff() function to return
        n_rois and structured metadata.

        Returns
        -------
        list
            A list of metadata dictionaries for all ROIs.
        """

        self.n_rois, metadata = load_metadata_from_tiff(self)
        self.roi_list = np.arange(self.n_rois)

        return metadata

    def _load_data(self):
        """
        Load and process imaging data.

        Uses load_imaging_data_from_tiff() function
        to load the raw .tiff data and perform some
        initial processing, including:
        - gating-PMT artifact correction 
        - 3D median filter 

        Returns
        -------
        list
            A list of processed imaging data arrays.
        """

        return load_imaging_data_from_tiff(self)

    def __len__(self):
        return len(self.roi_list)

    def __getattr__(self, name: str):
        return self.__dict__[f"_{name}"]

    def __setattr__(self, name: str, value):
        self.__dict__[f"_{name}"] = value

    def __getitem__(self, roi_index: int) -> None:
        if roi_index not in self.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self.roi_list)}')

        return self.get_roi(roi_index)

    @cache
    def get_roi(self, roi_index: int) -> object:
        """
        Retrieve a cached instance of the specified ROI.

        Parameters
        ----------
        roi_index : int
            Index of the ROI to retrieve.

        Returns
        -------
        Roi
            The ROI object corresponding to the specified index.
        """

        roi_metadata = self.metadata[roi_index]

        z_ind = roi_metadata['z_ind']
        start_index = sum(self.n_sweeps[:z_ind])
        n_sweeps_for_roi = roi_metadata['n_sweeps']
        end_index = start_index + n_sweeps_for_roi

        return Roi(
            roi_index,
            roi_metadata,
            self.data[start_index:end_index])


class Roi:
    """
    Representation of imaging data of a single Region of Interest (ROI).

    It takes the loaded z-layer data and crops out the corresponding ROI.
    This class encapsulates the data and metadata for an individual ROI.
    It also supports accessing individual sweeps as specialized sweep objects.
    """

    def __init__(
            self,
            roi_index: int,
            roi_metadata: dict,
            sweeps_data: np.ndarray,
    ) -> None:
        """
        Initialize an ROI instance.

        Parameters
        ----------
        roi_index : int
            Index of the ROI in the dataset.
        roi_metadata : dict
            Metadata dictionary containing information about the ROI, such as
            size, position, and acquisition settings.
        sweeps_data : np.ndarray
            Raw imaging data for the sweeps corresponding to this ROI, from
            which the ROI will be cropped.

        Returns
        -------
        None
        """

        self.roi_index = roi_index

        self.roi_metadata = roi_metadata
        roi_bounds = self.roi_metadata['roi_bounds']
        pixel_resolution_xy = self.roi_metadata['pixel_resolution_xy']

        # Crop out the corresponding roi
        self.roi = [data[
            :, :, roi_bounds[0]:roi_bounds[1], :int(pixel_resolution_xy[0])]
            for data in sweeps_data]

        # Set roi metadata as attributes
        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        self.sweep_list = np.arange(self.roi_metadata['n_sweeps'])

    @property
    def shape(self):
        """
        Get the shape of the ROI's data.

        Returns
        -------
        str
            The shape of the ROI's data as a formatted string.
            Format: "(number_of_sweeps, dimensions_of_frames)"
        """
        return f"({len(self.roi)}, {[n for n in self.roi[0].shape]})"

    def __getitem__(self, sweep_index: int) -> object:

        if sweep_index not in self.sweep_list:
            raise IndexError(f'Sweep {sweep_index} not in {self.sweep_list}')

        return self.get_sweep(sweep_index)

    @cache
    def get_sweep(self, sweep_index: int) -> object:
        """
        Retrieve a cached instance of the sweep object for the specified index.

        Parameters
        ----------
        sweep_index : int
            Index of the sweep to retrieve.

        Returns
        -------
        SweepCA3 or SweepBLA
            A specialized sweep object, either `SweepCA3` or `SweepBLA`,
            depending on the ADC channel associated with the sweep.
        """

        adc = self.adc_list[sweep_index]

        # FIXME: This is a temporary solution to differentiate between
        # the two types of stimulation. A more robust solution should be implemented.

        if adc == 'IN 3':  # IN3 corresponds to the electrode digital output
            return SweepCA3(
                sweep_index,
                self.roi_metadata,
                self.roi[sweep_index])

        elif adc == 'IN 2':  # IN2 corresponds to the LED light digital output
            return SweepBLA(
                sweep_index,
                self.roi_metadata,
                self.roi[sweep_index])


class Sweep:
    """
    Generic class for Sweep, parent to SweepCA3 and SweepBLA.
    Representation of a single sweep for a specific ROI.

    A sweep corresponds to a sequence of imaging frames (single trial)
    acquired for a single ROI across all active channels.
    This class provides methods to visualize,
    manipulate, and save sweep data, as well as access individual channels.
    """

    def __init__(
            self,
            sweep_index: int,
            roi_metadata: dict,
            sweep_data: np.array,
    ) -> None:
        """
        Initialize a Sweep instance.

        Parameters
        ----------
        sweep_index : int
            Index of the sweep within the ROI.
        roi_metadata : dict
            Metadata dictionary containing ROI information and acquisition
            parameters.
        sweep_data : np.ndarray
            The raw frame sequence data for the sweep.

        Returns
        -------
        None
        """

        self.sweep_index = sweep_index
        self.roi_metadata = roi_metadata

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        self.sweep = sweep_data

    @property
    def shape(self):
        """
        Get the shape of the sweep data.

        Returns
        -------
        tuple
            Shape of the sweep data (frames, channels, height, width).
        """
        return self.sweep.shape

    def show(
        self,
        timestamps: bool = False,
        norm: list or tuple or np.ndarray = None
    ) -> None:
        """
        Visualize the sweep frame sequence in a real-time animation.

        Frames for all active channels are displayed side-by-side in a CV2
        window. Press 'Q' to exit the animation.

        Parameters
        ----------
        timestamps : bool, optional
            If True, displays the elapsed time on the frames. Default is False.
        norm : list | tuple | np.ndarray, optional
            Normalization range for the frame intensity values. Default is None.

        Returns
        -------
        None
        """

        return animate_frames(
            self.sweep,
            self.frame_rate,
            active_channels=self.ch_active,
            timestamps=timestamps,
            norm=norm)

    def save(
            self,
            output_file: str or Path = None,
            timestamps: bool = False,
            norm: list or tuple or np.ndarray = None
    ) -> None:
        """
        Save the sweep data as a video file.
        Exploits OpenCV tools.

        Parameters
        ----------
        output_file : str | Path, optional
            Path to save the video file. If None, saves with the sweep's filename.
        timestamps : bool, optional
            If True, includes timestamps in the video.
        norm : list | tuple | np.ndarray, optional
            Normalization range for the frame intensity values. Default is None.

        Returns
        -------
        None
        """

        if not output_file:
            output_file = self.filename.with_suffix('.avi')

        print(f'Saved! at {output_file}')

        return save_frames(
            self.sweep,
            self.frame_rate,
            output_file,
            active_channels=self.ch_active,
            timestamps=timestamps,
            norm=norm)

    @property
    def ch1(self) -> object:
        """
        Access the data for channel 1 of the sweep.

        Returns
        -------
        Channel
            A `Channel` object containing the data and metadata for channel 1.
        """

        channel_index = 0
        channel_data = self.sweep[:, channel_index, :, :]

        return Channel(
            channel_index,
            self.roi_metadata,
            channel_data)

    @property
    def ch2(self) -> object:
        """
        Access the data for channel 2 of the sweep.

        Returns
        -------
        Channel
            A `Channel` object containing the data and metadata for channel 2.
        """

        channel_index = 1
        channel_data = self.sweep[:, channel_index, :, :]

        return Channel(
            channel_index,
            self.roi_metadata,
            channel_data)


class SweepCA3(Sweep):
    """
    Specialized Sweep class for CA3 region data.

    Inherits from the `Sweep` class without modification. This is used for
    clearer differentiation of sweeps belonging to the CA3 region.
    """

    pass


class SweepBLA(Sweep):
    """
    Specialized Sweep class for BLA region data.

    Inherits from the `Sweep` class without modification. This is used for
    clearer differentiation of sweeps belonging to the BLA region.
    """

    pass


class Channel:
    """
    Representation of a single imaging channel for a sweep.

    The `Channel` class encapsulates the data and metadata for a specific channel 
    in a single sweep. It provides methods for visualization, saving, and 
    statistical analysis of the channel data, as well as access to individual frames.
    """

    def __init__(
            self,
            channel_index,
            roi_metadata,
            channel_data,
    ) -> None:
        """
        Initialize a `Channel` instance.

        Parameters
        ----------
        channel_index : int
            Index of the channel within the sweep (0-based).
        roi_metadata : dict
            Metadata dictionary containing ROI and acquisition parameters.
        channel_data : np.ndarray
            Frame sequence for the channel, with shape (n_frames, height, width).

        Returns
        -------
        None
        """

        self.channel_index = channel_index
        self.roi_metadata = roi_metadata
        self.channel = channel_data

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

    def __getitem__(
            self,
            frame_index: int
    ) -> object:
        """
        When indexed, returns a single Frame.
        """
        frame_data = self.channel[frame_index]

        return Frame(
            frame_index,
            self.roi_metadata,
            frame_data)

    @property
    def shape(self):
        """
        Get the shape of the channel data.

        Returns
        -------
        tuple
            Shape of the channel data (frames, height, width).
        """
        return self.channel.shape

    def show(
        self,
        timestamps: bool = False,
        norm: list or tuple or np.ndarray = None
    ) -> None:
        """
        Visualize the frame sequence for the channel.
        Exploits OpenCV tools.

        Parameters
        ----------
        timestamps : bool, optional
            If True, displays elapsed time on each frame. Default is False.
        norm : list | tuple | np.ndarray, optional
            Normalization range for frame intensity values. Default is None.

        Returns
        -------
        None
            Displays the frame sequence in an OpenCV window. Press 'Q' to exit.
        """

        return animate_frames(
            self.channel,
            self.frame_rate,
            active_channels=None,
            timestamps=timestamps,
            norm=norm)

    def save(
            self,
            output_file: str or Path,
            timestamps: bool = False,
            norm: list or tuple or np.ndarray = None
    ) -> None:
        """
        Save the frame sequence of the channel as a video file.
        Exploits OpenCV tools.

        Parameters
        ----------
        output_file : str | Path
            Path to save the video file.
        timestamps : bool, optional
            If True, includes elapsed time on the frames. Default is False.
        norm : list | tuple | np.ndarray, optional
            Normalization range for frame intensity values. Default is None.

        Returns
        -------
        None
        """

        if not output_file:
            output_file = self.filename.with_suffix('.avi')

        print(f'Saved! at {output_file}')

        return save_frames(
            self.channel,
            self.frame_rate,
            output_file,
            active_channels=None,
            timestamps=timestamps,
            norm=norm)

    @property
    def maxproj(self) -> object:
        """
        Compute the maximum intensity projection of the channel.

        Returns
        -------
        Frame
            A `Frame` object containing the maximum projection of the channel.
        """

        frame_data = np.max(self.channel, axis=0)

        return Frame(
            'maxproj',
            self.roi_metadata,
            frame_data)

    @property
    def std(self) -> object:
        """
        Compute the standard deviation projection of the channel.

        Returns
        -------
        Frame
            A `Frame` object containing the standard deviation projection of the channel.
        """

        frame_data = np.std(self.channel, axis=0)

        return Frame(
            'std',
            self.roi_metadata,
            frame_data)


class Frame:
    """
    Representation of a single frame.

    The `Frame` class encapsulates data and metadata for a single imaging frame 
    within a specific ROI, channel, and sweep. It provides methods for 
    visualization, saving, and applying basic image processing techniques.
    """

    def __init__(
            self,
            frame_index: int or str,
            roi_metadata: dict,
            frame_data: np.ndarray
    ) -> None:
        """
        Initialize a `Frame` instance.

        Parameters
        ----------
        frame_index : int or str
            Index or identifier for the frame.
        roi_metadata : dict
            Metadata dictionary containing information about the ROI, channel, 
            and sweep.
        frame_data : np.ndarray
            The raw image data for the frame.

        Returns
        -------
        None
        """

        self.frame_index = frame_index
        self.roi_metadata = roi_metadata
        self.frame = frame_data

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

    def show(
            self,
            ax: plt.Axes = None,
            cmap: str = None,
            norm: list or tuple or np.ndarray = None,
            show_cmap_bar: bool = False,
            hide_xticks: bool = False,
            hide_yticks: bool = False,
    ) -> None:
        """
        Visualize the frame using Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        cmap : str, optional
            Colormap for the image. Default is None.
        norm : tuple | list | np.ndarray, optional
            Normalization range as [min, max]. Default is None.
        show_cmap_bar : bool, optional
            If True, display a colorbar alongside the image. Default is False.
        hide_xticks : bool, optional
            If True, hide x-axis tick marks. Default is False.
        hide_yticks : bool, optional
            If True, hide y-axis tick marks. Default is False.

        Returns
        -------
        None
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if norm and isinstance(norm, (list, tuple, np.ndarray)):
            norm = Normalize(vmin=norm[0], vmax=norm[1])

        im = ax.imshow(self.frame, cmap=cmap, norm=norm)

        if show_cmap_bar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if hide_xticks:
            ax.set_xticks([])
        if hide_yticks:
            ax.set_yticks([])

    def save(
            self,
            output_file: str or Path = None,  # type: ignore
            data_type: str = None,
            norm: tuple or list or np.ndarray = None
    ) -> None:
        """
        Save the frame as a `.tif` file.

        Parameters
        ----------
        output_file : str | Path, optional
            Path to save the frame. If None, saves with the frame's default filename.
        data_type : str, optional
            Data type to save the frame as. Default is None (uses original data type).
        norm : tuple | list | np.ndarray, optional
            Normalization range as [min, max]. Default is None.

        Returns
        -------
        None
        """

        if not output_file:
            output_file = self.filename.with_suffix('.tif')

        print(f'Saved! at {output_file}')

        if not data_type:
            data_type = self.data_type

        return save_single_frame(
            frame=self.frame,
            output_file=output_file,
            data_type=data_type,
            norm=norm)

    @property
    def median(self, kernel: int = 3):
        """
        Compute a median-filtered version of the frame.

        Parameters
        ----------
        kernel : int, optional
            Size of the median filter kernel. Default is 3.

        Returns
        -------
        np.ndarray
            Median-filtered frame.

        Note
        ------
        Images are already filtered at the source using a 3d median filter.
        """

        return median(self.frame, size=kernel)

    @property
    def gauss(self):
        """
        Compute a Gaussian-smoothed version of the frame.

        Returns
        -------
        np.ndarray
            Gaussian-smoothed frame.
        """

        return gaussian(self.frame)
