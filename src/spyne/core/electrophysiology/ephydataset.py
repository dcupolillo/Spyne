""" Created on Thu Feb 13 11:08:11 2025
    @author: dcupolillo """

from __future__ import annotations
from pathlib import Path
import flammkuchen as fl
from tqdm import tqdm
import numpy as np
from functools import cache
import tensorflow as tf
from spyne.core.electrophysiology.config import EphyDatasetConfig
from spyne.core.electrophysiology.load import (
    load_metadata_from_abf, load_data_from_abf, find_test_pulse_window)
from spyne.core.electrophysiology.io import save_metadata
from spyne.core.electrophysiology.analysis.pyabf_passive_props import (
    analyze_I_steps, unpack_spike_dfs, passive_properties)
from spyne.core.electrophysiology.analysis.synaptic_events import (
    BLA_EPSCs,
    CA3_EPSCs,
    detect_all_events,
    sEPSCs,
)
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
            model_fn: str | Path = None,
            recording_channel: int = None,
            scaling: float = None,
            unit: str = None,
            win_size: int = None,
            direction: str = None,
            bla_epoch_idx: int = None,
            ca3_epoch_idx: int = None,
            model_threshold: float = None,
            batch_size: int = None,
            peak_w: float = None,
            rel_prom_cutoff: float = None,
            convolve_window: int = None,
            gradient_convolve_window: int = None,
            resample_to_600: bool = None,
            _force_recompute: bool = False,
            **config_overrides
    ) -> None:
        """
        Initialize the EphyDataset with an ImagingDataset instance.

        Parameters
        ----------
        dataset : ImagingDataset
            An instance of ImagingDataset containing imaging data and metadata.
        model_fn : str | Path, optional
            Path to the miniML event detection model. Overrides config value.
        _force_recompute : bool, optional
            If True, forces recomputation of all data, ignoring cached results.

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
        self._force_recompute = _force_recompute

        # Initialize configuration
        self.config = EphyDatasetConfig(
            model_fn=model_fn,
            recording_channel=recording_channel,
            scaling=scaling,
            unit=unit,
            win_size=win_size,
            direction=direction,
            bla_epoch_idx=bla_epoch_idx,
            ca3_epoch_idx=ca3_epoch_idx,
            model_threshold=model_threshold,
            batch_size=batch_size,
            peak_w=peak_w,
            rel_prom_cutoff=rel_prom_cutoff,
            convolve_window=convolve_window,
            gradient_convolve_window=gradient_convolve_window,
            resample_to_600=resample_to_600,
            **config_overrides)

        self.abf_file_list = self._dataset.abf_file_list

        _I_step_folder = self._dataset.folder / "raw"
        self.Istep_filename = [
            fname for fname in _I_step_folder.glob("*.abf")
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

            "sweep_x": "processed/electrophysiology/sweep_x.h5",
            "sweep_y": "processed/electrophysiology/sweep_y.h5",
            "sweep_cmd": "processed/electrophysiology/sweep_cmd.h5",
            "sweep_scanner": "processed/electrophysiology/sweep_scanner.h5",
            "sweep_stim": "processed/electrophysiology/sweep_stim.h5",
            "sweep_led": "processed/electrophysiology/sweep_led.h5",
            "sweep_pmtgate": "processed/electrophysiology/sweep_pmtgate.h5",
            "Ih": "analysis/electrophysiology/Ih.h5",
            "Ra": "analysis/electrophysiology/Ra.h5",
            "Rm": "analysis/electrophysiology/Rm.h5",
            "Iss": "analysis/electrophysiology/Iss.h5",
            "rise_time": "analysis/electrophysiology/rise_time.h5",
            "decay_time": "analysis/electrophysiology/decay_time.h5",
            "tau_sec": "analysis/electrophysiology/tau_sec.h5",
            "BLA_EPSCs": "analysis/electrophysiology/BLA_EPSCs.h5",
            "BLA_EPSCs_x": "analysis/electrophysiology/BLA_EPSCs_x.h5",
            "BLA_amplitudes": "analysis/electrophysiology/BLA_amplitudes.h5",
            "CA3_EPSCs": "analysis/electrophysiology/CA3_EPSCs.h5",
            "CA3_EPSCs_x": "analysis/electrophysiology/CA3_EPSCs_x.h5",
            "CA3_amplitudes": "analysis/electrophysiology/CA3_amplitudes.h5",
            "sEPSCs": "analysis/electrophysiology/sEPSCs.h5",
            "sEPSCs_x": "analysis/electrophysiology/sEPSCs_x.h5",
            "sEPSC_amplitudes": "analysis/electrophysiology/sEPSC_amplitudes.h5",
            "sEPSC_counts": "analysis/electrophysiology/sEPSC_counts.h5",
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
        metadata_filename = (
            self._dataset.folder / "processed" / "electrophysiology" / metadata_filename
        )
        
        try:
            metadata = fl.load(metadata_filename)

            for key, value in metadata.items():
                if not hasattr(self, key):
                    setattr(self, key, [])
                getattr(self, key).append(value)

            return metadata
        
        except Exception as e:
            pass
        
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
            (self._dataset.folder / "processed" / "electrophysiology" / "metadata.h5")
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
            path = self._dataset.folder

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
        self._collect_synaptic_events(save, save_path=output_folder)
        self._collect_Isteps(save, save_path=output_folder)

    def _collect_synaptic_events(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """Collect BLA, CA3, and spontaneous EPSCs from one detection pass.

        For each ABF file, `detect_all_events` is run only once. BLA-evoked,
        CA3-evoked, and spontaneous events are then extracted from the same
        precomputed event list.
        """

        n_files = len(self.abf_file_list)

        self.BLA_EPSCs = np.empty(n_files, dtype=object)
        self.BLA_EPSCs_x = np.empty(n_files, dtype=object)
        self.BLA_amplitudes = np.empty(n_files, dtype=object)

        self.CA3_EPSCs = np.empty(n_files, dtype=object)
        self.CA3_EPSCs_x = np.empty(n_files, dtype=object)
        self.CA3_amplitudes = np.empty(n_files, dtype=object)

        self.sEPSCs = np.empty(n_files, dtype=object)
        self.sEPSCs_x = np.empty(n_files, dtype=object)
        self.sEPSC_amplitudes = np.empty(n_files, dtype=object)
        self.sEPSC_counts = np.empty(n_files, dtype=object)

        model = tf.keras.models.load_model(self.model_fn)

        for n, file_path in tqdm(
                enumerate(self.abf_file_list),
                desc="Collecting synaptic events",
                total=n_files):

            all_events = detect_all_events(self, file_path, model=model)

            (bla_events, bla_events_x, bla_amplitudes, _, _, _, _, _, _, _,
             _, _, _, _, _, _) = BLA_EPSCs(
                self,
                file_path,
                all_events=all_events)

            (ca3_events, ca3_events_x, ca3_amplitudes, _, _, _, _, _, _, _,
             _, _, _, _, _, _) = CA3_EPSCs(
                self,
                file_path,
                all_events=all_events)

            spontaneous_by_sweep = sEPSCs(
                self,
                file_path,
                all_events=all_events)

            self.BLA_EPSCs[n] = bla_events
            self.BLA_EPSCs_x[n] = bla_events_x
            self.BLA_amplitudes[n] = bla_amplitudes

            self.CA3_EPSCs[n] = ca3_events
            self.CA3_EPSCs_x[n] = ca3_events_x
            self.CA3_amplitudes[n] = ca3_amplitudes

            n_sweeps = len(spontaneous_by_sweep)
            file_events = np.empty(n_sweeps, dtype=object)
            file_events_x = np.empty(n_sweeps, dtype=object)
            file_amplitudes = np.empty(n_sweeps, dtype=object)
            file_counts = np.zeros(n_sweeps, dtype=int)

            for sweep_n, sweep_events in enumerate(spontaneous_by_sweep):
                file_events[sweep_n] = sweep_events["events"]
                file_events_x[sweep_n] = sweep_events["events_x"]
                file_amplitudes[sweep_n] = sweep_events["amplitudes"]
                file_counts[sweep_n] = sweep_events["events"].shape[0]

            self.sEPSCs[n] = file_events
            self.sEPSCs_x[n] = file_events_x
            self.sEPSC_amplitudes[n] = file_amplitudes
            self.sEPSC_counts[n] = file_counts

        self.BLA_EPSCs = np.stack(self.BLA_EPSCs, axis=0)
        self.BLA_EPSCs_x = np.stack(self.BLA_EPSCs_x, axis=0)
        self.BLA_amplitudes = np.stack(self.BLA_amplitudes, axis=0)

        self.CA3_EPSCs = np.stack(self.CA3_EPSCs, axis=0)
        self.CA3_EPSCs_x = np.stack(self.CA3_EPSCs_x, axis=0)
        self.CA3_amplitudes = np.stack(self.CA3_amplitudes, axis=0)

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
                "CA3_EPSCs": self.CA3_EPSCs,
                "CA3_EPSCs_x": self.CA3_EPSCs_x,
                "CA3_amplitudes": self.CA3_amplitudes,
                "sEPSCs": self.sEPSCs,
                "sEPSCs_x": self.sEPSCs_x,
                "sEPSC_amplitudes": self.sEPSC_amplitudes,
                "sEPSC_counts": self.sEPSC_counts,
            }

            for key, value in data_to_save.items():
                fl.save(saving_folder / f"{key}.h5", value)

    def _collect_timeseries(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """
        """
        saving_folder = (
            self._dataset.folder
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
                filename = self._files_mapping[key]
                fl.save(saving_folder / filename, value)
                print(f"Saved {key} to {saving_folder / filename}")

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
        """Backward-compatible wrapper for unified synaptic collection."""

        self._collect_synaptic_events(save=save, save_path=save_path)

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

    def _collect_sEPSCs(
            self,
            save: bool,
            save_path: str or Path = None
    ) -> None:
        """Backward-compatible wrapper for unified synaptic collection."""

        self._collect_synaptic_events(save=save, save_path=save_path)

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
