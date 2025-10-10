""" Created on Fri Aug  2 14:58:52 2024
    @author: dcupolillo """

from __future__ import annotations
import pyabf
import numpy as np
from tqdm import tqdm
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spyne.core.electrophysiology.ephydataset import EphyDataset


adc_dict = {
    'IN 0': 'patch',
    'IN 2': 'led',
    'IN 3': 'electrode',
    'IN 4': 'pmt_gate',
    'IN 5': 'scanner',
    }

adc_channels_idx = {
    'patch': 0,
    'led': 1,
    'electrode': 2,
    'pmt_gate': 3,
    'scanner': 4,
    'photodiode': 5,
}


def load_metadata_from_abf(ephy_instance: EphyDataset) -> list:
    """
    Load metadata from ABF files associated with the EphyDataset.
    
    Parameters
    ----------
    ephy_instance : EphyDataset
        An instance of the EphyDataset class.
    Returns
    -------
    list
        A list of metadata dictionaries for each ABF file, 
        each containing information such as:
        - Data type
        - Channel count
        - Sampling rate
        - Data points per ms
        - Number of sweeps
        - Sweep duration
        - Protocol name
        - Z-index and Z-coordinate
        - Coplanar ROIs
        - Channel list
        - ADC names and units
        - ADC mapping
        - Epoch start and end times
    Notes
    -----
    - Ensures unique ABF files are processed to avoid redundant metadata entries.
    - Maps ADC names to more descriptive labels using a predefined dictionary.
    - Associates each ABF file with its corresponding Z-index and coplanar ROIs
        from the linked ImagingDataset.
    """

    unique_file_list = list(
        {file_path.parent: file_path
            for file_path in ephy_instance.abf_file_list}.values())
    
    dataset_instance = ephy_instance._dataset

    metadata = [None] * len(unique_file_list)

    for file_n in tqdm(
            range(len(unique_file_list)), desc="Loading metadata"):

        abf_file = unique_file_list[file_n]
        abf = pyabf.ABF(abf_file)

        metadata[file_n] = dict({
            "dtype": abf._dtype,
            "channel_count": abf.channelCount,
            "sampling_rate": abf.dataRate,
            "data_sec_per_point": abf.dataSecPerPoint,
            "data_points_per_ms": abf.dataPointsPerMs,
            "n_sweeps": abf.sweepCount,
            "sweep_duration": abf.sweepLengthSec,
            "protocol": abf.protocol,
            "z_ind": dataset_instance[file_n].roi_metadata["z_ind"],
            "z": dataset_instance[file_n].roi_metadata["z"],
            "coplanar_rois":
                dataset_instance[file_n].roi_metadata["coplanar_roi_n"],
            "channel_list": abf.channelList,
            "adc_names": abf.adcNames,
            "adc_units": abf.adcUnits,
            "adc": [adc_dict.get(name, name) for name in abf.adcNames],
            "epochs_start": abf.sweepEpochs.p1s,
            "epochs_end": abf.sweepEpochs.p2s,
        })

    return metadata


def load_data_from_abf(ephy_instance: EphyDataset):

    # Determine number of files, sweeps, and points from first file
    n_files = len(ephy_instance.abf_file_list)
    abf_test = pyabf.ABF(ephy_instance.abf_file_list[0])
    n_sweeps = abf_test.sweepCount
    n_points = abf_test.sweepPointCount

    sweep_y = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)
    sweep_x = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)
    sweep_cmd = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)
    sweep_scanner = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)
    sweep_stim = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)
    sweep_led = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)
    sweep_pmtgate = np.empty((n_files, n_sweeps, n_points), dtype=np.float32)

    for file_n in tqdm(
            range(len(ephy_instance.abf_file_list)),
            desc="Loading data"):

        abf_file = ephy_instance.abf_file_list[file_n]
        abf = pyabf.ABF(abf_file)

        for sweep_n in abf.sweepList:

            for n_channel in range(len(abf.adcNames)):

                channel = abf.adcNames[n_channel]
                channel_label = adc_dict.get(channel, None)
                channel_index = adc_channels_idx.get(channel_label, None)

                if channel_index is None:
                    continue

                abf.setSweep(sweep_n, channel=channel_index)

                if channel not in adc_dict.keys():
                    continue
                
                if adc_dict[channel] == 'patch':
                    sweep_y[file_n, sweep_n, :] = abf.sweepY
                    sweep_x[file_n, sweep_n, :] = abf.sweepX
                    sweep_cmd[file_n, sweep_n, :] = abf.sweepC

                elif adc_dict[channel] == 'scanner':
                    sweep_scanner[file_n, sweep_n, :] = abf.sweepY

                elif adc_dict[channel] == 'stim':
                    sweep_stim[file_n, sweep_n, :] = abf.sweepY

                elif adc_dict[channel] == 'led':
                    sweep_led[file_n, sweep_n, :] = abf.sweepY

                elif adc_dict[channel] == 'pmt':
                    sweep_pmtgate[file_n, sweep_n, :] = abf.sweepY

    return (
        sweep_x,
        sweep_y,
        sweep_cmd,
        sweep_scanner,
        sweep_stim,
        sweep_led,
        sweep_pmtgate
    )

        
def get_digital_output_list(abf_path):

    abf = pyabf.ABF(abf_path)
    adc_names = abf.adcNames
    adc_list = []

    for sweep_n in range(abf.sweepCount):

        events = []

        for n, adc in enumerate(adc_names):

            if (adc == 'IN 0' or adc == 'IN 5' or adc == 'IN 4'):
                continue

            abf.setSweep(sweep_n, channel=n)

            # Find indices where the digital output is triggered
            trigger_indices = np.where(abf.sweepY > 1)[0]

            # Append events with the corresponding times and channels
            for idx in trigger_indices:
                events.append((abf.sweepX[idx], adc))

        # Sort events by time to maintain chronological order
        events.sort(key=lambda x: x[0])

        sweep_adcs = []
        for event in events:
            adc = event[1]
            if adc not in sweep_adcs:
                sweep_adcs.append(adc)

        adc_list.extend(sweep_adcs)

    return adc_list

def find_test_pulse_window(
        abf_path: str or Path,
) -> tuple:
    """
    Find the start and end times of the test pulse in an ABF file.

    Parameters
    ----------
    abf_path : str or Path
        Path to the ABF file.

    Returns
    -------
    tuple
        A tuple containing the start and end times of the test pulse in seconds.
    """

    abf = pyabf.ABF(abf_path)
    adc_names = abf.adcNames

    if 'IN 0' not in adc_names:
        raise ValueError("Channel 'IN 0' not found in ADC names.")

    abf.setSweep(0, channel=adc_names.index('IN 0'))

    epochs_starts = abf.sweepEpochs.p1s
    epochs_ends = abf.sweepEpochs.p2s
    epochs_levels = abf.sweepEpochs.levels

    # find index of positive level
    max_level = max(epochs_levels)
    epoch_index = epochs_levels.index(max_level)

    test_pulse_start_time = epochs_starts[epoch_index]
    test_pulse_end_time = epochs_ends[epoch_index]

    return max_level, test_pulse_start_time, test_pulse_end_time
