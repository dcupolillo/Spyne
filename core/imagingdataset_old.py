""" Created on Mon Oct 30 13:59:21 2023
    @author: dcupolillo """

from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from functools import cache

import ROIpy as rp
from spyne.display.plot_frames import (
    animate_frames, save_frames, save_single_frame)
from spyne.core.utils.movie_utils import (
    simplify_attr_name, simplify_key)
from spyne.core.utils.filters import median, gaussian
from spyne.core.utils.pyabf_adc import get_digital_output_list, get_pmt_gate
from neuronpath.path import NeuronPath
import logging

logging.basicConfig(level=logging.DEBUG)


class ImagingDataset:

    def __init__(
            self,
            paths: NeuronPath
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

        self.folder = paths.imaging
        self.parent_folder = paths.parent
        self.file_list = paths.tif_file_list
        self.abf_file_list = paths.abf_file_list

        self.sf = rp.Scanfields(paths)

        self.rect_periods = [rect.rectangle_period
                             for zplane in self.sf.neuComp
                             for rect in zplane]

        self.coplanar_n = [rect.z_ind
                           for zplane in self.sf.neuComp
                           for rect in zplane]

        self.unique_roifile_list = list(
            {file_path.parent: file_path
             for file_path in self.file_list}.values())

        self.z_values = sorted(
            {int(file_path.parent.name[1:])
             for file_path in self.file_list})
        z_index_map = {z: idx for idx, z in enumerate(self.z_values)}

        self.grouped_sweeps = [[] for _ in range(len(self.z_values))]

        for file_path in self.file_list:

            z_value = int(file_path.parent.name[1:])

            z_index = z_index_map[z_value]
            self.grouped_sweeps[z_index].append(file_path)

        self.n_sweeps = [len(n) for n in self.grouped_sweeps]

        # Retrieve metadata
        self.roi_metadata = []
        n_roi = 0

        for n, file_path in enumerate(self.unique_roifile_list):

            with tifffile.TiffFile(file_path) as tif:
                scanimage_metadata = tif.scanimage_metadata
                frame_data = scanimage_metadata['FrameData']
                roigroup_data = (
                    scanimage_metadata['RoiGroups']['imagingRoiGroup'])
                rois = roigroup_data['rois']

                if isinstance(rois, dict):
                    rois = [rois]

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

                    self.roi_metadata.append({
                        'affine': scanfield['affine'],
                        'translate': translate,
                        'center_xy': center_xy,
                        'center_xy_ref': center_xy_ref,
                        'pixel_resolution_xy': pixel_resolution_xy,
                        'pixel_to_ref_transform': (
                            scanfield['pixelToRefTransform']),
                        'roi_uuid': scanfield['roiUuid'],
                        'roi_uuid_int64': scanfield['roiUuiduint64'],
                        'rotation_degrees': scanfield['rotationDegrees'],
                        'size_xy': scanfield['sizeXY'],
                        'resolution': resolution,
                        'z': z,
                        'z_ind': n,
                        'n_roi': n_roi,
                        'n_sweeps': len(
                            [name
                             for name in file_path.parent.iterdir()
                             if name.suffix.lower() in ('.tif', '.tiff')]),
                        'frame_rate': frame_data[
                            'SI.hRoiManager.scanFrameRate'],
                        'rectangle_period': self.rect_periods[n_roi],
                        'flyto': (
                            frame_data['SI.hScan2D.flytoTimePerScanfield']),
                        'flyback': frame_data['SI.hScan2D.flybackTimePerFrame']
                    })
                    n_roi += 1

            if n == 0:
                for key, value in frame_data.items():
                    attr_name = simplify_key(key)
                    setattr(self, attr_name, value)

                # Simpler attribute names for convenience
                metadata = simplify_attr_name(frame_data)
                for key, value in metadata.items():
                    attr_name = simplify_key(key)
                    setattr(self, attr_name, value)

                self.frames_shape = tifffile.imread(file_path).shape
                self.frames_dtype = tifffile.imread(file_path).dtype

        self.n_rois = len(self.roi_metadata)
        self.roi_list = np.arange(self.n_rois)
        self.frame_list = np.arange(0, self.n_frames)

        grouped_metadata = [[] for _ in range(len(self.z_values))]

        for metadata in self.roi_metadata:
            grouped_metadata[metadata['z_ind']].append(metadata)

        for group in grouped_metadata:
            start_y = 0
            for metadata in group:
                end_y = start_y + metadata['pixel_resolution_xy'][1]
                metadata['roi_bounds'] = [start_y, end_y]
                start_y = end_y

        self.roi_metadata = [
            metadata
            for group in grouped_metadata
            for metadata in group]

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

    def __getitem__(self, roi_index: int) -> object:

        if roi_index not in self.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self.roi_list)}')

        return self.get_roi(roi_index)

    def __len__(self):
        return len(self.roi_list)

    @cache
    def get_roi(self, roi_index: int) -> object:
        return Roi(self.paths, roi_index)


class Roi(ImagingDataset):

    def __init__(
            self,
            paths: NeuronPath,
            roi_index: int
    ) -> None:

        super().__init__(paths)

        self.roi_index = roi_index

        # Set roi metadata as attributes
        for key, value in self.roi_metadata[self.roi_index].items():
            setattr(self, key, value)

        self.sweep_list = np.arange(self.n_sweeps)
        self.sweep_file_list = self.grouped_sweeps[self.z_ind]

        self.abf_file = self.abf_file_list[self.z_ind]
        self.adc_list = get_digital_output_list(self.abf_file)

        self.pmtgate_window = get_pmt_gate(self.abf_file)

    @property
    def shape(self):
        return [self.size_xy[0], self.size_xy[1]]

    def __getitem__(self, sweep_index: int) -> object:

        if sweep_index not in self.sweep_list:
            raise IndexError(f'Sweep {sweep_index} not in {self.sweep_list}')

        return self.get_sweep(sweep_index)

    @cache
    def get_sweep(self, sweep_index: int) -> object:
        adc = self.adc_list[sweep_index]
        if adc == 'IN 3':
            return Sweep_CA3(self.paths, self.roi_index, sweep_index)
        elif adc == 'IN 2':
            return Sweep_BLA(self.paths, self.roi_index, sweep_index)


class Sweep(Roi):

    def __init__(
            self,
            paths: NeuronPath,
            roi_index: int,
            sweep_index: int
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

        super().__init__(
            paths,
            roi_index)

        self.sweep_index = sweep_index
        self.sweep_file = self.sweep_file_list[self.sweep_index]

        frames = tifffile.imread(self.sweep_file)
        roi_bounds = self.roi_metadata[self.roi_index]['roi_bounds']
        start_height, end_height = map(int, roi_bounds)

        self.frame_seq = frames[
            :, :, start_height:end_height, :int(self.pixel_resolution_xy[0])]

    @property
    def shape(self):
        return self.frames.shape

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
            self.frame_seq,
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
            self.frame_seq,
            self.frame_rate,
            output_file,
            active_channels=self.ch_active,
            timestamps=timestamps,
            norm=norm)

    @property
    def ch1(self) -> object:

        channel_index = 0

        return Channel(
            self.paths,
            self.roi_index,
            self.sweep_index,
            self.frame_seq,
            channel_index)

    @property
    def ch2(self) -> object:

        channel_index = 1

        return Channel(
            self.paths,
            self.roi_index,
            self.sweep_index,
            self.frame_seq,
            channel_index)


class Sweep_CA3(Sweep):
    pass


class Sweep_BLA(Sweep):

    def __init__(
            self,
            paths: NeuronPath,
            roi_index: int,
            sweep_index: int
    ) -> None:

        super().__init__(
            paths,
            roi_index,
            sweep_index)

        roi_on_z = [
            roi['n_roi']
            for roi in self.roi_metadata
            if roi['z_ind'] == self.z_ind]
        print(roi_on_z)

        if roi_index == roi_on_z[0] or roi_index == roi_on_z[-1]:

            frame_index_to_replace = 16 if roi_index == roi_on_z[0] else 15
            prevoius_frame_index = frame_index_to_replace - 1
            following_frame_index = frame_index_to_replace + 1

            self.frame_with_artifact = (
                self.frame_seq[frame_index_to_replace].copy())
            previous_frame = self.frame_seq[prevoius_frame_index]
            following_frame = self.frame_seq[following_frame_index]

            # Calculate the average frame between frame 15 and frame 17
            average_frame = np.mean(
                [previous_frame, following_frame], axis=0)

            # Replace frame 16 with the average frame
            self.frame_seq[frame_index_to_replace] = average_frame


class Channel(Sweep):

    def __init__(
            self,
            paths: NeuronPath,
            roi_index: int,
            sweep_index: int,
            frame_seq: np.array,
            channel_index: int
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

        super().__init__(
            paths,
            roi_index,
            sweep_index)

        self.channel_index = channel_index

        self.channel_seq = frame_seq[:, self.channel_index, :, :]

    def __getitem__(
            self,
            frame_index: int
    ) -> object:
        """
        When indexed, returns a single Frame.
        """

        return Frame(
            self.paths,
            self.roi_index,
            self.sweep_index,
            self.channel_seq,
            self.channel_index,
            frame_index)

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
            self.channel_seq,
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
            self.channel_seq,
            self.frame_rate,
            output_file,
            timestamps=timestamps,
            norm=norm)

    @property
    def maxproj(self) -> object:

        return Frame(
            self.paths,
            self.roi_index,
            self.sweep_index,
            self.frame_seq,
            self.channel_index,
            'maxproj')

    @property
    def std(self) -> object:

        return Frame(
            self.paths,
            self.roi_index,
            self.sweep_index,
            self.frame_seq,
            self.channel_index,
            'std')


class Frame(Channel):

    def __init__(
            self,
            paths: NeuronPath,
            roi_index: int,
            sweep_index: int,
            frame_seq: np.array,
            channel_index: int,
            frame_index: int or str
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

        super().__init__(
            paths,
            roi_index,
            sweep_index,
            frame_seq,
            channel_index)

        self.frame_index = frame_index

        if self.frame_index == 'maxproj':
            self.frame = np.max(self.channel_seq, axis=0)

        elif self.frame_index == 'std':
            self.frame = np.std(self.channel_seq, axis=0)

        else:
            self.frame = self.channel_seq[self.frame_index]

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
