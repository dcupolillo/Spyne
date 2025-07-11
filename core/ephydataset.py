""" Created on Thu Feb 13 11:08:11 2025
    @author: dcupolillo """

import numpy as np
from functools import cache
from spyne.core.utils.load import (
    load_metadata_from_abf, load_abf, load_epochs)
from spyne.core.utils.abf_utils import (
    filter_sweep_bessel, analyze_test_pulse, calculate_holding_current,
    EPSC_analysis)


class EphyDataset:

    def __init__(
            self,
            dataset: object,
    ) -> None:

        if type(dataset).__name__ != 'ImagingDataset':
            raise TypeError('Invalid input type for dataset')

        self.dataset = dataset
        self.abf_file_list = self.dataset.abf_file_list

        z_values = sorted(
            {int(file_path.parent.name[1:])
                for file_path in self.dataset.file_list})
        z_index_map = {idx: z for idx, z in enumerate(z_values)}

        # To reproduce ImagingDataset structure, I want to index
        # each roi individually. This implies that multiple rois
        # share the same ephy recording if colplanar
        self.roi_recording_map = {
            n: z_index_map[roi['z_ind']]
            for n, roi in enumerate(self.dataset.metadata)}

        self.metadata = self._load_metadata()
        self.data = self._load_data()
        self.epochs = self._load_epochs()

        self.collect_all_data()
        self.BLA_EPSCs()

    def _load_metadata(self) -> list:

        metadata_list = load_metadata_from_abf(self)

        for metadata in metadata_list:
            for key, value in metadata.items():
                if not hasattr(self, key):
                    setattr(self, key, [])
                getattr(self, key).append(value)

        return metadata_list

    def _load_data(self) -> dict:

        data_dict = load_abf(self)

        for key, value in data_dict.items():
            setattr(self, key, value)

        return data_dict

    def _load_epochs(self) -> list:

        return load_epochs(self)

    def collect_all_data(self):

        self.Ih = []
        self.Ra = []
        self.Rm = []
        self.Iss = []

        self.test_pulse_Vm = []
        self.test_pulse_ts = []

        for z, zlayer in enumerate(self.data["sweepy"]):

            I_holding = calculate_holding_current(self, z, 9)

            (
                Ra,
                Rm,
                Iss,
                test_pulse_Vm,
                test_pulse_ts
            ) = analyze_test_pulse(self, z, 2)

            self.Ih.append(I_holding)
            self.Ra.append(Ra)
            self.Rm.append(Rm)
            self.Iss.append(Iss)
            self.test_pulse_Vm.append(test_pulse_Vm)
            self.test_pulse_ts.append(test_pulse_ts)

    def BLA_EPSCs(
            self,
            BLA_EPSC_length: float = 0.2
    ) -> None:
        """
        Extracts EPSC properties for all recordings
        and dynamically stores them as instance attributes.

        Parameters:
        ----------
        BLA_EPSC_length : float, optional
            The length of the EPSC segment to analyze (in seconds).
            Default is 0.2 sec.
        """

        # Initialize a dictionary to store EPSC analysis results
        BLA_EPSC_data = {
            "amplitudes_BLA": [],
            "charges_BLA": [],
            "onsets_BLA": [],
            "onsets_ts_BLA": [],
            "onsets_sec_BLA": [],
            "offsets_BLA": [],
            "offsets_ts_BLA": [],
            "offsets_sec_BLA": [],
            "peaks_BLA": [],
            "peaks_ts_BLA": [],
            "peaks_sec_BLA": [],
            "EPSCs_BLA": [],
            "EPSCs_ts_BLA": [],
        }

        # Process each z-layer (recording)
        for z, zlayer in enumerate(self.data["sweepy"]):
            epsc_results = EPSC_analysis(self, z, BLA_EPSC_length)

            # Dynamically populate the dictionary
            # while adding input type to the keys
            for key in epsc_results.keys():
                attr_name = f"{key}_BLA"
                BLA_EPSC_data[attr_name].append(epsc_results[key])

        # Set the dictionary keys as instance attributes
        for key, value in BLA_EPSC_data.items():
            setattr(self, key, value)

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
            key: value[recording_index] for key, value in self.data.items()}
        roi_epochs = self.epochs[recording_index]

        roi_Ih = self.Ih[recording_index]
        roi_Ra = self.Ra[recording_index]
        roi_Rm = self.Rm[recording_index]
        roi_Iss = self.Iss[recording_index]
        roi_test_pulse_Vm = self.test_pulse_Vm[recording_index]
        roi_test_pulse_ts = self.test_pulse_ts[recording_index]

        return RoiEphy(
            roi_index,
            recording_index,
            roi_metadata,
            roi_data,
            roi_epochs,
            roi_Ih,
            roi_Ra,
            roi_Rm,
            roi_Iss,
            roi_test_pulse_Vm,
            roi_test_pulse_ts)


class RoiEphy:

    def __init__(
            self,
            roi_index: int,
            recording_index: int,
            roi_metadata: dict,
            roi_data: dict,
            roi_epochs: dict,
            roi_Ih: np.ndarray,
            roi_Ra: np.ndarray,
            roi_Rm: np.ndarray,
            roi_Iss: np.ndarray,
            roi_test_pulse_Vm: np.ndarray,
            roi_test_pulse_ts: np.ndarray
    ) -> None:

        self.roi_index = roi_index
        self.recording_index = recording_index
        self.metadata = roi_metadata
        self.data = roi_data
        self.epochs = roi_epochs

        self.Ih = roi_Ih
        self.Ra = roi_Ra
        self.Rm = roi_Rm
        self.Iss = roi_Iss
        self.test_pulse_Vm = roi_test_pulse_Vm
        self.test_pulse_ts = roi_test_pulse_ts

        self.digital_bits = [
            "scanning", "led", "pmt_gate", "electrode",
            None, None, None, None]

        for key, value in self.metadata.items():
            setattr(self, key, value)

        for key, value in self.data.items():
            setattr(self, key, value)

        for key, value in self.epochs.items():
            setattr(self, key, value)
