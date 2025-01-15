""" Created on Mon Oct 30 13:59:21 2023
    @author: dcupolillo """

from pathlib import Path
import numpy as np
import tifffile
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import accumulate
from functools import cache
import math

import ROIpy as rp
from spyne.display.plot_frames import (
    animate_frames, save_frames, save_single_frame)
from spyne.core.utils.movie_utils import (
    simplify_attr_name, simplify_key, print_scanimage_metadata)
from spyne.core.utils.filters import median, gaussian
from neuronpath.path import NeuronPath
import logging

logging.basicConfig(level=logging.DEBUG)


class ImagingDataset:

    def __init__(
            self,
            folder: str or Path or NeuronPath
    ) -> None:
        """
        Creates a Dataset parser for all dendritic ROIs of a single neuron.
        Neuronal data are organized in ROI groups corresponding
        to discrete z planes. ROI groups include a number of individual ROI,
        which are rotated rectangle objects.

        Hierarchical organization when indexed.
        Subclasses RoiGroup, Roi, Channel, Frame.

        Parameters
        ----------
        folder : str or Path or NeuroPath
            Path to folder containing images and recordings.

        Raises
        ------
        Exception
            if folder does not exist or is not a folder or is empty.

        Returns
        -------
        None.

        Example usage
        -------
        dataset = ImagingDateset('path\\to\\folder')

        """

        # Ensure folder is a NeuronPath
        folder_path = NeuronPath(folder)

        if not folder_path.is_dir():
            raise Exception("Path must be a path to a FOLDER.")

        if not folder_path.exists():
            raise Exception(f"{folder} does not exist")

        if not any(folder_path.iterdir()):
            raise Exception(f"{folder} is empty")

        self.folder = folder_path
        self.parent_folder = self.folder.parent

        self.scanfields_bundle = rp.Scanfields(
            NeuronPath(self.folder.parent).stackpath,
            NeuronPath(self.folder.parent).tracepath).neuComp

        # Collects all imaging files and their parent folders
        self.file_list, self.scanfield_folder_list = zip(*[
            (file_path, file_path.parent)
            for file_path in self.folder.rglob('*')
            if file_path.suffix.lower() in ('.tif', '.tiff')
        ])

        # Sort them based on numerical nomenclature
        self.scanfield_folder_list = sorted(list(set(
            self.scanfield_folder_list)),
            key=lambda path: int(path.name[-2:]))

        # Scanfield-related attributes
        self.scanfield_list = np.arange(0, len(self.scanfield_folder_list))
        self.scanfield_ind_list = np.array(
            [int(path.name[-2:])
             for path in self.scanfield_folder_list])
        self.n_scanfields = len(self.scanfield_folder_list)

    def __getattr__(
            self,
            name: str):
        logging.debug(f"Accessing attribute: {name}")
        try:
            return self.__dict__[f"_{name}"]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has"
                f"no attribute '{name}'")

    def __setattr__(
            self,
            name: str,
            value):
        self.__dict__[f"_{name}"] = value

    def __getitem__(
            self,
            scanfield_index: int
    ) -> object:
        """
        When indexed, returns a Scanfield as a group of coplanar ROIs.
        """
        if scanfield_index not in self.scanfield_list:
            raise IndexError(
                f'Scanfield {scanfield_index} not in {self.scanfield_list}')
        return self.get_roigroup(scanfield_index)

    def __len__(self):
        return len(self.scanfield_list)

    @cache
    def get_roigroup(
            self,
            scanfield_index: int
    ) -> None:
        """
        Cached retrieval of RoiGroup instances based on scanfield_index.
        """
        return RoiGroup(self.folder, scanfield_index)


class RoiGroup(ImagingDataset):

    def __init__(
            self,
            dataset_folder: str,
            scanfield_index: int
    ) -> None:
        """
        Collections of Rois laying on the same z plane.
        All sweeps / single scanfield / all ROIs / all channels
        Inherits from ImagingDataset
        Subclasses Roi when indexed.

        Parameters
        ----------
        dataset_folder : str
            Path to dataset folder.
        scanfield_index : int
            The 0-based index of the z planes.

        Returns
        -------
        None

        """

        super().__init__(dataset_folder)

        self.scanfield_index = scanfield_index
        self.sweeps_folder = self.scanfield_folder_list[self.scanfield_index]

        sweeps_list = [
            file_path
            for file_path in self.sweeps_folder.glob('*')
            if file_path.suffix.lower() in ('.tif', '.tiff')
        ]

        self.sweep_list = np.arange(0, len(sweeps_list))
        self.n_sweeps = len(self.sweep_list)

        self.load_metadata()

    def load_metadata(self) -> None:
        """
        Access Scanimage metadata from a file within the scanfield folder.
        Creates class attributes describing the scanfield.
        """

        folder = self.folder / self.sweeps_folder

        # Get filename of first sweep
        # Assumes all sweeps files share the same metadata
        for file_path in folder.iterdir():
            if file_path.suffix.lower() in ('.tif', '.tiff'):
                self._filename = file_path
                break

        # Access metadata and split them into frame_data and roigroup_data
        with tifffile.TiffFile(self._filename) as tif:
            scanimage_metadata = tif.scanimage_metadata
            frame_data = scanimage_metadata['FrameData']
            roigroup_data = scanimage_metadata['RoiGroups']['imagingRoiGroup']

        # Process and set frame attributes
        for key, value in frame_data.items():
            attr_name = simplify_key(key)
            setattr(self, attr_name, value)

        # Simpler attribute names for convenience
        metadata = simplify_attr_name(frame_data)
        for key, value in metadata.items():
            attr_name = simplify_key(key)
            setattr(self, attr_name, value)

        self.frame_list = np.arange(0, self.n_frames)

        # Process and set roi group attributes
        for key, value in roigroup_data.items():
            if key != 'rois':
                attr_name = f'roigroup_{simplify_key(key)}'
                setattr(self, attr_name, value)
            else:
                attr_name = simplify_key(key)
                setattr(self, attr_name, value)

        self.n_roi = (
            1 if isinstance(roigroup_data['rois'], dict)
            else len(roigroup_data['rois']))

        self.roi_list = np.arange(0, self.n_roi)

        if isinstance(self.rois, dict):
            self.rois = [self.rois]

        roi_heights = [roi['scanfields'].get('pixelResolutionXY', [0, 0])[1]
                       for roi in self.rois]
        self.roi_widths = [
            roi['scanfields'].get('pixelResolutionXY', [0, 0])[0]
            for roi in self.rois]
        self.cumulative_heights = list(accumulate([0] + roi_heights))

    @cache
    def get_roi(
            self,
            roi_index: int
    ) -> None:
        """
        Cached retrieval of Roi instances based on roi_index.
        """
        return Roi(self.folder,
                   self.scanfield_index,
                   roi_index)

    def __getitem__(
            self,
            roi_index: int
    ) -> 'Roi':
        """
        When indexed, returns an individual ROI.
        """
        if roi_index not in self.roi_list:
            raise IndexError(f'Roi {roi_index} not in {self.roi_list}')
        return self.get_roi(roi_index)

    def __len__(self):
        return self.n_roi


class Roi(RoiGroup):

    def __init__(
            self,
            dataset_folder: str,
            scanfield_index: int,
            roi_index: int
    ) -> None:
        """
        2D cross section of a ROI at a particular z plane.
        Representation of a Frame sequence
        All sweep / single ROI / all channels
        Inherits from Scanfield

        Parameters
        ----------
        dataset_folder : str
            path to dataset folder.
        scanfield_index : int
            0-based index of the z planes
        roi_index : int
            0-based index of Roi within the scanfield

        Returns
        -------
        None

        """

        super().__init__(
            dataset_folder,
            scanfield_index)

        self.roi_index = roi_index

        # Update sweeps_list specific to this ROI instance
        sweeps_list = [
            file_path
            for file_path in self.sweeps_folder.glob('*')
            if file_path.suffix.lower() in ('.tif', '.tiff')
        ]

        self.sweep_list = np.arange(0, len(sweeps_list))
        self.n_sweeps = len(self.sweep_list)

        self.get_single_scanfield_params()

        for key, value in self.rois[self.roi_index].items():
            setattr(self, key, value)

    def get_single_scanfield_params(self) -> None:

        single_roi = self.rois[self.roi_index]['scanfields']
        for key, attr in single_roi.items():
            attr_name = f'{simplify_key(key)}'
            setattr(self, attr_name, attr)

        # Update roiuuid based on matching center_xy
        # Looking through the scanfields bundle
        print(self.roiuuid)
        for z_plane in self.scanfields_bundle:
            for roi in z_plane:
                roi_center = roi.center_deg
                dataset_center = self.centerxy

                # Consider approximation
                if math.isclose(
                        roi_center[0], dataset_center[0], rel_tol=1e-9) and \
                   math.isclose(
                       roi_center[1], dataset_center[1], rel_tol=1e-9):
                    print(roi.roi_uuid)
                    self.roiuuid = roi.roi_uuid
                    print(self.roiuuid)


        self.resolution = [
            self.sizexy[0] / self.pixelresolutionxy[0],
            self.sizexy[1] / self.pixelresolutionxy[1]
            ]

        self.centerxy_ref = np.dot(
            [0.5, 0.5], np.array(self.affine).T[:-1, :])

        self.translate = np.array(
            [[1, 0, self.centerxy[0] - self.centerxy_ref[0]],
             [0, 1, self.centerxy[1] - self.centerxy_ref[1]],
             [0, 0, 1]]
            )

        self.z = float(self.rois[0]['name'].split(' = ')[-1].split(', ')[0])

    @property
    def shape(self):
        return (self.width, self.height)

    @cache
    def get_sweep(
            self,
            sweep_index: int
    ) -> None:
        """
        Cached retrieval of Roi instances based on roi_index.
        """
        return Sweep(
            self.folder,
            self.scanfield_index,
            self.roi_index,
            sweep_index)

    def __getitem__(
            self,
            sweep_index: int
    ) -> object:
        """
        When indexed, returns an individual Sweep.
        """

        if sweep_index not in self.sweep_list:
            raise IndexError(f'Sweep {sweep_index} not in {self.sweep_list}')

        return self.get_sweep(sweep_index)

    def __repr__(self):
        return print_scanimage_metadata(self, self.roi_index)


class Sweep(Roi):

    def __init__(
            self,
            dataset_folder: str,
            scanfield_index: int,
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
            dataset_folder,
            scanfield_index,
            roi_index)

        self.sweep_index = sweep_index

        frames = None

        for path in self.sweeps_folder.iterdir():
            if re.match(r'^.*\.tif(f?)$', path.name):
                match = re.search(r'(\d+)\.tif(f?)$', path.name)
                if match:
                    last_digit = int(match.group(1))
                    if last_digit == self.sweep_index + 1:
                        frames = tifffile.imread(path)
                        break
        if frames is None:
            raise FileNotFoundError(
                f"Sweep file for index {self.sweep_index} not found in "
                f"{self.sweeps_folder}")

        self.frame_seq = frames[
            :,
            :,
            self.cumulative_heights[self.roi_index]:
                self.cumulative_heights[self.roi_index + 1],
            : self.roi_widths[self.roi_index]]

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
            self.folder,
            self.scanfield_index,
            self.roi_index,
            self.sweep_index,
            channel_index)

    @property
    def ch2(self) -> object:

        channel_index = 1

        return Channel(
            self.folder,
            self.scanfield_index,
            self.roi_index,
            self.sweep_index,
            channel_index)


class Channel(Sweep):

    def __init__(
            self,
            dataset_folder: str,
            scanfield_index: int,
            roi_index: int,
            sweep_index: int,
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
            dataset_folder,
            scanfield_index,
            roi_index,
            sweep_index)

        self.channel_index = channel_index
        self.channel_seq = self.frame_seq[:, self.channel_index, :, :]

    def __getitem__(
            self,
            frame_index: int
    ) -> object:
        """
        When indexed, returns a single Frame.
        """

        return Frame(
            self.folder,
            self.scanfield_index,
            self.roi_index,
            self.sweep_index,
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
            self.folder,
            self.scanfield_index,
            self.roi_index,
            self.sweep_index,
            self.channel_index,
            'maxproj')

    @property
    def std(self) -> object:

        return Frame(
            self.folder,
            self.scanfield_index,
            self.roi_index,
            self.sweep_index,
            self.channel_index,
            'std')


class Frame(Channel):

    def __init__(
            self,
            dataset_folder: str,
            scanfield_index: int,
            roi_index: int,
            sweep_index: int,
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
            dataset_folder,
            scanfield_index,
            roi_index,
            sweep_index,
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
