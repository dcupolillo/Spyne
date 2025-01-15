""" Created on Mon Oct 30 13:59:21 2023
    @author: dcupolillo """

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from functools import cache
from scipy.ndimage import median_filter
from skimage.util import img_as_uint
import skimage

import ROIpy as rp
from spyne.display.plot_frames import (
    animate_frames, save_frames, save_single_frame)
from spyne.core.utils.movie_utils import rows_deviation, modify_frames
from spyne.core.utils.filters import median, gaussian
from spyne.core.utils.pyabf_adc import get_digital_output_list, get_pmt_gate
from neuronpath.path import NeuronPath
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)


class ImagingDataset:

    def __init__(
            self,
            paths: NeuronPath,
            kernel_size: tuple = (3, 3, 3)
    ) -> None:

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

        self.paths = paths
        self.folder = paths.imaging
        self.parent_folder = paths.parent
        self.file_list = paths.tif_file_list
        self.abf_file_list = paths.abf_file_list
        self.median_filter_kernel_size = kernel_size

        self.sf = rp.Scanfields(paths)

        self.metadata = self.load_metadata()
        self.data = self.load_data()

    def load_metadata(self):

        metadata: list(dict) = []

        rect_periods = [rect.rectangle_period
                        for zplane in self.sf.neuComp
                        for rect in zplane]

        unique_roifile_list = list(
            {file_path.parent: file_path
             for file_path in self.file_list}.values())

        z_values = sorted(
            {int(file_path.parent.name[1:])
             for file_path in self.file_list})
        z_index_map = {z: idx for idx, z in enumerate(z_values)}

        grouped_sweeps = [[] for _ in range(len(z_values))]

        for file_path in self.file_list:

            z_value = int(file_path.parent.name[1:])
            z_index = z_index_map[z_value]
            grouped_sweeps[z_index].append(file_path)

        self.n_sweeps = [len(n) for n in grouped_sweeps]

        self.coplanar_n = [rect.z_ind
                           for zplane in self.sf.neuComp
                           for rect in zplane]

        # Retrieve metadata
        n_roi = 0
        coplanar_dict = {}

        for n, file_path in enumerate(
                tqdm(unique_roifile_list, desc="Loading metadata")):

            abf_file = self.abf_file_list[n]
            adc_list = get_digital_output_list(abf_file)

            with tifffile.TiffFile(file_path) as tif:
                scanimage_metadata = tif.scanimage_metadata
                frame_data = scanimage_metadata['FrameData']
                roigroup_data = (
                    scanimage_metadata['RoiGroups']['imagingRoiGroup'])
                rois = roigroup_data['rois']

                rois = [rois] if isinstance(rois, dict) else rois

                for roi in rois:
                    z = float(roi['name'].split(",")[0].split(" = ")[-1])
                    scanfield = roi['scanfields']
                    center_xy = scanfield['centerXY']
                    size_xy = scanfield['sizeXY']
                    pixel_resolution_xy = scanfield['pixelResolutionXY']
                    center_xy_ref = np.dot(
                        [0.5, 0.5],
                        np.array(scanfield['affine']).T[:-1, :])
                    translate = np.array(
                        [[1, 0, center_xy[0] - center_xy_ref[0]],
                         [0, 1, center_xy[1] - center_xy_ref[1]],
                         [0, 0, 1]])
                    resolution = [
                        size_xy[0] / pixel_resolution_xy[0],
                        size_xy[1] / pixel_resolution_xy[1]]

                    if z not in coplanar_dict:
                        coplanar_dict[z] = []
                    coplanar_dict[z].append(n_roi)

                    metadata.append({
                        'objective resolution':
                            frame_data["SI.objectiveResolution"],
                        'affine': scanfield['affine'],
                        'translate': translate,
                        'center_xy': center_xy,
                        'center_xy_ref': center_xy_ref,
                        'pixel_resolution_xy': pixel_resolution_xy,
                        'pixel_to_ref_transform':
                            scanfield['pixelToRefTransform'],
                        'roi_uuid': scanfield['roiUuid'],
                        'roi_uuid_int64': scanfield['roiUuiduint64'],
                        'rotation_degrees': scanfield['rotationDegrees'],
                        'size_xy': scanfield['sizeXY'],
                        'resolution': resolution,
                        'z': z,
                        'z_ind': n,
                        'n_roi': n_roi,
                        'n_sweeps': self.n_sweeps[n],
                        'n_frames':
                            frame_data['SI.hStackManager.framesPerSlice'],
                        'frame_rate':
                            frame_data['SI.hRoiManager.scanFrameRate'],
                        'rectangle_period': rect_periods[n_roi],
                        'flyto':
                            frame_data['SI.hScan2D.flytoTimePerScanfield'],
                        'flyback':
                            frame_data['SI.hScan2D.flybackTimePerFrame'],
                        'adc_list': [adc for adc in adc_list],
                        'ch_active': frame_data['SI.hChannels.channelsActive'],
                        'LUT': frame_data['SI.hChannels.channelLUT'],
                        'offset': frame_data['SI.hChannels.channelOffset'],
                        'subtract_offset':
                            frame_data['SI.hChannels.channelSubtractOffset'],
                        'input_range':
                            frame_data['SI.hChannels.channelInputRange']
                    })

                    n_roi += 1

        for meta in metadata:
            z = meta['z']
            meta['coplanar_roi_n'] = coplanar_dict[z]

        grouped_metadata = [[] for _ in range(len(z_values))]

        for meta in metadata:
            grouped_metadata[meta['z_ind']].append(meta)

        for group in grouped_metadata:
            start_y = 0
            for meta in group:
                end_y = start_y + meta['pixel_resolution_xy'][1]
                meta['roi_bounds'] = [start_y, end_y]
                start_y = end_y

        self.n_rois = len(metadata)
        self.roi_list = np.arange(self.n_rois)

        return [meta
                for group in grouped_metadata
                for meta in group]

    def load_data(self):

        data = [None] * len(self.file_list)

        for n, file_path in enumerate(tqdm(
                self.file_list, desc="Loading data")):

            # raw data are dtype signed int16
            raw_frames = tifffile.imread(file_path)

            # Negative values clipped as 0
            # FIXME: fix the range
            frames = raw_frames.copy()
            # frames = img_as_uint(frames)

            # Correct gating-PMT black stripe
            # Dectect deviating rows
            deviation_channel1 = rows_deviation(
                frames[:, 0, :, :], threshold_factor=3, plot=False)
            deviation_channel2 = rows_deviation(
                frames[:, 1, :, :], threshold_factor=3, plot=False)

            deviating_rows = {}

            for frame in set(deviation_channel1.keys()).union(
                    set(deviation_channel2.keys())):

                rows1 = deviation_channel1.get(frame, [])
                rows2 = deviation_channel2.get(frame, [])
                deviating_rows[frame] = sorted(set(rows1 + rows2))

            sorted_deviating_rows = dict(sorted(deviating_rows.items()))

            consecutive_deviating_rows = {}
            frame_numbers = list(sorted_deviating_rows.keys())

            for frame_number in frame_numbers:
                consecutive_deviating_rows[frame_number] = (
                    sorted_deviating_rows[frame_number])

            modified_frames = modify_frames(
                frames, consecutive_deviating_rows)

            # Apply 3D median filter
            filtered_frames = modified_frames.copy()
            filtered_frames[:, 0, :, :] = median_filter(
                filtered_frames[:, 0, :, :],
                size=(3, 3, 3),
                mode='wrap')
            filtered_frames[:, 1, :, :] = median_filter(
                filtered_frames[:, 1, :, :],
                size=(3, 3, 3),
                mode='wrap')

            data[n] = modified_frames

        return data

    def __getattr__(self, name: str):
        logging.debug(f"Accessing attribute: {name}")
        try:
            return self.__dict__[f"_{name}"]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has"
                f"no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self.__dict__[f"_{name}"] = value

    def __getitem__(self, roi_index: int) -> None:

        if roi_index not in self.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self.roi_list)}')

        return self.get_roi(roi_index)

    def __len__(self):
        return len(self.roi_list)

    @cache
    def get_roi(self, roi_index: int) -> object:

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

    def __init__(
            self,
            roi_index: int,
            roi_metadata: dict,
            sweeps_data: np.ndarray,
    ) -> None:

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
        return f"({len(self.roi)}, {[n for n in self.roi[0].shape]})"

    def __getitem__(self, sweep_index: int) -> object:

        if sweep_index not in self.sweep_list:
            raise IndexError(f'Sweep {sweep_index} not in {self.sweep_list}')

        return self.get_sweep(sweep_index)

    @cache
    def get_sweep(self, sweep_index: int) -> object:
        adc = self.adc_list[sweep_index]
        if adc == 'IN 3':
            return SweepCA3(
                sweep_index,
                self.roi_metadata,
                self.roi[sweep_index])
        elif adc == 'IN 2':
            return SweepBLA(
                sweep_index,
                self.roi_metadata,
                self.roi[sweep_index])


class Sweep:

    def __init__(
            self,
            sweep_index: int,
            roi_metadata: dict,
            sweep_data: np.array,
    ) -> None:
        """
        Representation of a Frame sequence
        Single sweep / single ROI / all channels
        Inherit from Roi

        Parameters
        ----------
        dataset_folder : str
            path to dataset folder.
        scanfield_index : int
            0-based index of the z planes
        roi_index : int
            0-based index of Roi within the scanfield
        sweep_index : int
            0-based index of sweep.

        Returns
        -------
        None
        """

        self.sweep_index = sweep_index
        self.roi_metadata = roi_metadata

        # Set roi metadata as attributes
        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

        self.sweep = sweep_data

    @property
    def shape(self):
        return self.sweep.shape

    def show(
            self,
            timestamps: bool = False,
            pulse: bool = False,
            norm: list or tuple or np.ndarray = None):
        """
        Shows the frame sequence of all channels
        side by side in a cv2 window.
        Press Q to exit.

        Parameters
        ----------
        timestamps : bool, optional
            If True, elapsed time is displayed.
        pulse : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

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

        # filename = self.filename.stem
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

        channel_index = 0
        channel_data = self.sweep[:, channel_index, :, :]

        return Channel(
            channel_index,
            self.roi_metadata,
            channel_data)

    @property
    def ch2(self) -> object:

        channel_index = 1
        channel_data = self.sweep[:, channel_index, :, :]

        return Channel(
            channel_index,
            self.roi_metadata,
            channel_data)


class SweepCA3(Sweep):
    pass


class SweepBLA(Sweep):
    pass

    # def __init__(
    #         self,
    #         sweep_index: int,
    #         roi_metadata: dict,
    #         sweep_data: np.array,
    # ) -> None:

    #     super().__init__(sweep_index, roi_metadata, sweep_data)

    #     self.modify_sweep()

    # def modify_sweep(self):

    #     coplanar_rois = self.roi_metadata['coplanar_roi_n']
    #     roi_n = self.roi_metadata['n_roi']

    #     frame_to_modify = round(self.frame_rate / 1.00)

    #     # single roi in a plane case
    #     if len(coplanar_rois) == 1:
    #         self.modify_frame(frame_to_modify - 1)
    #         self.modify_frame(frame_to_modify)

    #     # multiple rois
    #     else:
    #         if roi_n == coplanar_rois[0]:
    #             self.modify_frame(frame_to_modify)

    #         elif roi_n == coplanar_rois[1]:
    #             self.modify_frame(frame_to_modify)

    #         elif roi_n == coplanar_rois[-1]:
    #             self.modify_frame(frame_to_modify - 1)

    # def modify_frame(self, frame_index):

    #     if 0 < frame_index < self.sweep.shape[0] - 1:
    #         previous_frame = self.sweep[frame_index - 1]
    #         following_frame = self.sweep[frame_index + 1]

    #         self.sweep[frame_index] = np.mean(
    #             [previous_frame, following_frame], axis=0)

    #         print(f"Frame {frame_index} modified")


class Channel:

    def __init__(
            self,
            channel_index,
            roi_metadata,
            channel_data,
    ) -> None:
        """
        Representation of a Frame sequence
        Single sweep / single ROI / one selected channels
        Inherit from Sweep

        Parameters
        ----------
        dataset_folder : str
            path to dataset folder.
        scanfield_index : int
            0-based index of the z planes.
        roi_index : int
            0-based index of Roi within the scanfield.
        sweep_index : int
            0-based index of sweep.
        channel_index : int
            0-based index of channel.

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
        return self.channel.shape

    def show(
            self,
            timestamps: bool = False,
            norm: list or tuple or np.ndarray = None):
        """
        Parameters
        ----------
        timestamps : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return animate_frames(
            self.channel,
            self.frame_rate,
            timestamps=timestamps,
            norm=norm)

    def save(
            self,
            output_file: str or Path,
            timestamps: bool = False,
            norm: list or tuple or np.ndarray = None
    ) -> None:
        """
        Parameters
        ----------
        timestamps : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not output_file:
            output_file = self.filename.with_suffix('.avi')
        print(f'Saved! at {output_file}')

        return save_frames(
            self.channel,
            self.frame_rate,
            output_file,
            timestamps=timestamps,
            norm=norm)

    @property
    def maxproj(self) -> object:

        frame_data = np.max(self.channel, axis=0)

        return Frame(
            'maxproj',
            self.roi_metadata,
            frame_data)

    @property
    def std(self) -> object:

        frame_data = np.std(self.channel, axis=0)

        return Frame(
            'std',
            self.roi_metadata,
            frame_data)


class Frame:

    def __init__(
            self,
            frame_index: int or str,
            roi_metadata: dict,
            frame_data: np.ndarray
    ) -> None:
        """
        Representation of a Single frame
        Single frame / single ROI / one selected channels / single sweep
        Inherit from Channel

        Parameters
        ----------
        dataset_folder : str
            DESCRIPTION.
        scanfield_index : int
            DESCRIPTION.
        roi_index : int
            DESCRIPTION.
        sweep_index : int
            DESCRIPTION.
        channel_index : int
            DESCRIPTION.
        frame_index : int or str
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """

        self.frame_index = frame_index
        self.roi_metadata = roi_metadata
        self.frame = frame_data

        for key, value in self.roi_metadata.items():
            setattr(self, key, value)

    def show(
            self,
            ax=None,
            cmap=None,
            norm=None,
            show_cmap_bar: bool = False,
            hide_xticks: bool = False,
            hide_yticks: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is None.
        norm : TYPE, optional
            DESCRIPTION. The default is None.
        show_cmap_bar : bool, optional
            DESCRIPTION. The default is False.
        hide_xticks : bool, optional
            DESCRIPTION. The default is False.
        hide_yticks : bool, optional
            DESCRIPTION. The default is False.
         : None

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
            output_file: str or Path = None,
            data_type: str = None,
            norm: tuple or list or np.ndarray = None
    ) -> None:

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
    def median(
            self,
            kernel=3):

        return median(self.frame, size=kernel)

    @property
    def gauss(self):

        return gaussian(self.frame)
