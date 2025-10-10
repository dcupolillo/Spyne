""" Created on Thu Feb 13 11:08:11 2025
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
import flammkuchen as fl
from tqdm import tqdm
import numpy as np
from functools import cache
import tensorflow as tf
from spyne.core.electrophysiology.load import (
    load_metadata_from_abf, load_data_from_abf, find_test_pulse_window)
from spyne.core.electrophysiology.io import save_metadata
from spyne.core.electrophysiology.analysis.pyabf_passive_props import (
    analyze_I_steps, unpack_spike_dfs, passive_properties)
from spyne.core.electrophysiology.analysis.synaptic_events import BLA_EPSCs
from spyne.core.imaging.imagingdataset import ImagingDataset


class EphyDataset:
    """
    Class to handle electrophysiology data associated with an ImagingDataset.
    Each ROI in the ImagingDataset is linked to its corresponding
    electrophysiology recording based on z-index. As a consequence,
    multiple ROIs may share the same electrophysiology recording if colplanar.
    """

    def __init__(
            self,
            dataset: ImagingDataset,
            model_fn: str or Path = r"C:\Users\dcupolillo\Projects\miniML\models\transfer_learning\NMJ_wt\lstm_transfer.h5",
    ) -> None:
        """
        Initialize the EphyDataset with an ImagingDataset instance.

        Parameters
        ----------
        dataset : ImagingDataset
            An instance of ImagingDataset containing imaging data and metadata.

        Raises
        ------
        TypeError
            If the provided dataset is not an instance of ImagingDataset.
        Exception
            If no ABF files are found in the dataset folder.
        """

        if not isinstance(dataset, ImagingDataset):
            raise TypeError('Invalid input type for dataset')

        self._dataset = dataset
        self.abf_file_list = self._dataset.abf_file_list

        I_step_folder = self._dataset.folder / "raw"
        self.Istep_filename = [
            fname for fname in I_step_folder.glob("*.abf")
        ][0]

        self.model_fn = model_fn

        if not self.abf_file_list:
            raise Exception(f"No ABF files found in {self._dataset.folder}.")

        z_values = sorted(
            {int(file_path.parent.name[1:])
                for file_path in self._dataset.file_list})
        z_index_map = {idx: z for idx, z in enumerate(z_values)}

        # To reproduce ImagingDataset structure, I want to index
        # each roi individually. This implies that multiple rois
        # share the same ephy recording if colplanar
        self.roi_recording_map = {
            n: z_index_map[roi['z_ind']]
            for n, roi in enumerate(self._dataset.metadata)}

        self.metadata = self._load_metadata()
        self._load_data()
        # self._data = self._load_data()

        # if self._data is not None:
        #     self.sweep_x, self.sweep_y, self.sweep_cmd, \
        #         self.sweep_scanner, self.sweep_stim, \
        #         self.sweep_led, self.sweep_pmtgate = self._data

        self.test_pulse_level, self.test_pulse_start, \
            self.test_pulse_end = find_test_pulse_window(self.abf_file_list[0])

    @property
    def data(self):
        """
        Ephy data for all z planes.

        Returns
        -------
        tuple
            A tuple containing processed electrophysiology data arrays."""

        return self._data

    @property
    def _files_mapping(self) -> dict:
        """
        Mapping of expected filenames for different data types.

        This property defines a mapping between attribute names
        and their corresponding expected filenames. It is used
        to check for the existence of these files in the dataset
        folder and load them if available.

        Returns
        -------
        dict
            A dictionary mapping attribute names to expected filenames.
        """
        # All .npy files now live in processed/electrophysiology
        return {
            "sweep_x": "sweep_x.npy",
            "sweep_y": "sweep_y.npy",
            "sweep_cmd": "sweep_cmd.npy",
            "sweep_scanner": "sweep_scanner.npy",
            "sweep_stim": "sweep_stim.npy",
            "sweep_led": "sweep_led.npy",
            "sweep_pmtgate": "sweep_pmtgate.npy",
            "Ih": "Ih.npy",
            "Ra": "Ra.npy",
            "Rm": "Rm.npy",
            "Iss": "Iss.npy",
            "rise_time": "rise_time.npy",
            "decay_time": "decay_time.npy",
            "tau_sec": "tau_sec.npy",
            "BLA_EPSCs": "BLA_EPSCs.npy",
            "BLA_EPSCs_x": "BLA_EPSCs_x.npy",
        }
    
    def load_metadata(
        self,
        filename: str or Path
    ) -> None:
        """
        Public method to load imaging metadata with different parameters.
        This method allows you to manually load imaging metadata.

        Parameters
        ----------
        filename : str or Path
            Filename to load metadata from.

        Returns
        -------
        None
        """
        self._metadata = self._load_metadata(metadata_filename=filename)

    def _load_metadata(
        self,
        metadata_filename: str or Path = "metadata.h5"
    ) -> list:
        """
        Load metadata for all z planes.

        This method extracts information from abf files.
        Uses load_metadata_from_abf() function to return
        structured metadata.

        Returns
        -------
        list
            A list of metadata dictionaries for each ABF file.
        """
        metadata_filename = self._dataset.folder / "processed" / "imaging" / metadata_filename
        metadata_list = load_metadata_from_abf(self)

        for metadata in metadata_list:
            for key, value in metadata.items():
                if not hasattr(self, key):
                    setattr(self, key, [])
                getattr(self, key).append(value)

        return metadata_list
    
    def save_metadata(
        self,
        save_path: Path = None,
        overwrite: bool = False
    ) -> None:
        """
        Save the ephy metadata to disk for fast loading later.

        This method saves the current metadata (self.metadata),
        allowing for much faster loading in subsequent sessions.

        Parameters
        ----------
        save_path : Path, optional
            Path where to save the metadata. If None, uses a standard
            filename in the dataset folder.
        overwrite : bool, optional
            Whether to overwrite existing file. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If metadata is not loaded yet.

        Example
        -------
        >>> dataset = ImagingDataset(folder)
        >>> ephy = EphyDataset(dataset)  # This processes the data
        >>> ephy.save_metadata()
        """

        save_path = (
            (self.folder / "processed" / "electrophy" / "metadata.h5")
            if save_path is None
            else save_path)

        save_metadata(self.metadata, save_path, overwrite)

    def _load_data(
            self,
            path: str or Path = None
    ) -> tuple:
        """
        Load precomputed data (if available) for faster analysis,
        or load and process electrophysiology data from ABF files.

        This method calls `load_data_from_abf`, which reads and processes
        all ABF files associated with the current `EphyDataset` instance.
        It extracts and organizes sweep data for all relevant analog/digital
        channels (e.g., patch, LED, electrode, PMT gate, scanner) and returns
        them as arrays suitable for downstream analysis.

        Parameters
        ----------
        path : str or Path, optional
            Path to the directory containing precomputed data files.
            If None, uses the dataset's parent directory.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing arrays for each extracted channel, typically:
                - sweep_x
                - sweep_y
                - sweep_cmd
                - sweep_scanner
                - sweep_stim
                - sweep_led
                - sweep_pmtgate
            Each array contains the corresponding channel's
            data for all sweeps.
        """
        # Use processed/electrophysiology as the default path for all .npy files
        if path is None:
            path = self._dataset.folder / "processed" / "electrophysiology"

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

    def collect_all_data(
            self,
            save: bool = True,
            save_path: str or Path = None
    ) -> None:
        """
        Process all ABF files to extract electrophysiology data,
        compute passive properties, and detect synaptic events.
        Optionally saves the processed data as .npy files.

        Parameters
        ----------
        save : bool, optional
            Whether to save the processed data as .npy files.
            Default is True.
        save_path : str or Path, optional
            Path to the directory where processed data should be saved.
            If None, saves to the dataset's parent directory.
        """
        output_folder = (
            self._dataset.folder / "processed" / "electrophysiology"
            if save_path is None else save_path)

        self._collect_timeseries(save, save_path=output_folder)
        self._collect_passive_properties(save, save_path=output_folder)
        self._collect_BLA_EPSCs(save, save_path=output_folder)
        self._collect_Isteps(save, save_path=output_folder)

    def _collect_timeseries(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """
        """
        saving_folder = (
            self._dataset.folder / "processed" / "electrophysiology"
            if save_path is None else save_path)

        self.sweep_x, self.sweep_y, self.sweep_cmd, \
            self.sweep_scanner, self.sweep_stim, \
            self.sweep_led, self.sweep_pmtgate = load_data_from_abf(self)

        if save:
            saving_folder = Path(saving_folder)
            saving_folder.mkdir(parents=True, exist_ok=True)
            
            
            data_to_save = {
                "sweep_x": self.sweep_x,
                "sweep_y": self.sweep_y,
                "sweep_cmd": self.sweep_cmd,
                "sweep_scanner": self.sweep_scanner,
                "sweep_stim": self.sweep_stim,
                "sweep_led": self.sweep_led,
                "sweep_pmtgate": self.sweep_pmtgate,
            }

            for key, value in data_to_save.items():
                fl.save(saving_folder / f"{key}.h5", value)

    def _collect_passive_properties(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """
        This method computes passive properties (Ih, Ra, Rm, Iss) for each ABF file,
        stores them as arrays structured by recording and sweep,
        and optionally saves them as .npy files.

        Parameters
        ----------
        save : bool
            Whether to save the computed properties as .npy files.
        save_path : str or Path, optional
            Path to the directory where properties should be saved.
            If None, saves to the dataset's parent directory.
        """

        (
            self.Ih,
            self.Ra,
            self.Rm,
            self.Iss,
            self.rise_time,
            self.decay_time,
            self.tau_sec
        ) = passive_properties(
            self.sweep_y,
            self.test_pulse_start,
            self.test_pulse_end,
            self.test_pulse_level)

        if save:

            saving_folder = (
                self._dataset.folder / "analysis" / "electrophysiology"
                if save_path is None else save_path)
            saving_folder = Path(saving_folder)
            saving_folder.mkdir(parents=True, exist_ok=True)
            
            data_to_save = {
                "Ih": self.Ih,
                "Ra": self.Ra,
                "Rm": self.Rm,
                "Iss": self.Iss,
                "rise_time": self.rise_time,
                "decay_time": self.decay_time,
                "tau_sec": self.tau_sec,
            }
            
            for key, value in data_to_save.items():
                fl.save(saving_folder / f"{key}.h5", value)

    def _collect_BLA_EPSCs(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """
        This method processes each ABF file to detect BLA EPSCs
        using the BLA_EPSCs function from the analysis module.
        
        It stores the detected events and their properties in arrays
        and optionally saves them as .npy files.
        """

        self.BLA_EPSCs = np.empty(len(self.abf_file_list), dtype=object)
        self.BLA_EPSCs_x = np.empty(len(self.abf_file_list), dtype=object)
        self.BLA_amplitudes = np.empty(len(self.abf_file_list), dtype=object)

        model = tf.keras.models.load_model(self.model_fn)

        for n, file_path in tqdm(enumerate(self.abf_file_list),
                desc="Collecting BLA EPSCs",
                total=len(self.abf_file_list)):

            (events, events_x, amplitudes, event_peak_values, peak_x,
             bsls_start_x, bsls_end_x, scores, tau, charges, risetimes,
             rise_min_x, rise_max_x, slopes, decaytimes, halfwidths
            ) = BLA_EPSCs(self, file_path, model=model)

            self.BLA_EPSCs[n] = events
            self.BLA_EPSCs_x[n] = events_x
            self.BLA_amplitudes[n] = amplitudes

        self.BLA_EPSCs = np.stack(self.BLA_EPSCs, axis=0)
        self.BLA_EPSCs_x = np.stack(self.BLA_EPSCs_x, axis=0)
        self.BLA_amplitudes = np.stack(self.BLA_amplitudes, axis=0)

        if save:
            saving_folder = (
                self._dataset.folder / "analysis" / "electrophysiology"
                if save_path is None else save_path)
            saving_folder = Path(saving_folder)
            saving_folder.mkdir(parents=True, exist_ok=True)
            
            data_to_save = {
                "BLA_EPSCs": self.BLA_EPSCs,
                "BLA_EPSCs_x": self.BLA_EPSCs_x,
                "BLA_amplitudes": self.BLA_amplitudes,
            }
            
            for key, value in data_to_save.items():
                fl.save(saving_folder / f"{key}.h5", value)

    def _collect_Isteps(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """
        This method computes I-V relationship and firing frequency
        from the I-step protocol ABF file. In addition, it unpacks
        spike features into separate attributes for easier access.
        """

        self.Isteps, self.IF, self.IV, self.spikes = analyze_I_steps(self.Istep_filename)
        (
            self.peaks_v, self.peaks_t,
            self.threshold_t, self.threshold_v,
            self.upstroke_t, self.upstroke_v,
            self.downstroke_t, self.downstroke_v,
            self.width, self.peak_amplitude
        ) = unpack_spike_dfs(self.spikes)

        if save:
            saving_folder = (
                self._dataset.folder / "processed" / "electrophysiology"
                if save_path is None else save_path)
            saving_folder = Path(saving_folder)
            saving_folder.mkdir(parents=True, exist_ok=True)
            
            data_to_save = {
                "Isteps": self.Isteps,
                "IF": self.IF,
                "IV": self.IV,
                "spikes": self.spikes,
                "peaks_v": self.peaks_v,
                "peaks_t": self.peaks_t,
                "threshold_t": self.threshold_t,
                "threshold_v": self.threshold_v,
                "upstroke_t": self.upstroke_t,
                "upstroke_v": self.upstroke_v,
                "downstroke_t": self.downstroke_t,
                "downstroke_v": self.downstroke_v,
                "width": self.width,
                "peak_amplitude": self.peak_amplitude,
            }
            
            for key, value in data_to_save.items():
                fl.save(saving_folder / f"{key}.h5", value)

    def __getitem__(self, roi_index: int) -> None:
        if roi_index not in self._dataset.roi_list:
            raise IndexError(
                f'Roi {roi_index} out of range {len(self.dataset.roi_list)}')
        return self.get_roi(roi_index)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self.dataset):
            roi_ephy = self.get_roi(self._current_index)
            self._current_index += 1
            return roi_ephy
        else:
            raise StopIteration

    @cache
    def get_roi(self, roi_index: int) -> object:

        recording_index = self.roi_recording_map[roi_index]

        roi_metadata = self.metadata[recording_index]
        
        roi_data = {
            "sweep_x": self.sweep_x[recording_index],
            "sweep_y": self.sweep_y[recording_index],
            "sweep_cmd": self.sweep_cmd[recording_index],
            "sweep_scanner": self.sweep_scanner[recording_index],
            "sweep_stim": self.sweep_stim[recording_index],
            "sweep_led": self.sweep_led[recording_index],
            "sweep_pmtgate": self.sweep_pmtgate[recording_index],
        }

        roi_Ih = self.Ih[recording_index]
        roi_Ra = self.Ra[recording_index]
        roi_Rm = self.Rm[recording_index]
        roi_Iss = self.Iss[recording_index]

        return RoiEphy(
            roi_index,
            recording_index,
            roi_metadata,
            roi_data,
            roi_Ih,
            roi_Ra,
            roi_Rm,
            roi_Iss)


class RoiEphy:

    def __init__(
            self,
            roi_index: int,
            recording_index: int,
            roi_metadata: dict,
            roi_data: dict,
            roi_Ih: np.ndarray,
            roi_Ra: np.ndarray,
            roi_Rm: np.ndarray,
            roi_Iss: np.ndarray,
    ) -> None:

        self.roi_index = roi_index
        self.recording_index = recording_index
        self.metadata = roi_metadata
        self.data = roi_data

        self.Ih = roi_Ih
        self.Ra = roi_Ra
        self.Rm = roi_Rm
        self.Iss = roi_Iss

        self.digital_bits = [
            "scanning", "led", "pmt_gate", "electrode",
            None, None, None, None]

        for key, value in self.metadata.items():
            setattr(self, key, value)

        # Unpack data
        for key, value in self.data.items():
            setattr(self, key, value)

        for key, value in self.epochs.items():
            setattr(self, key, value)
