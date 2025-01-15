""" Created on Fri Aug  2 14:58:52 2024
    @author: dcupolillo """

import pyabf
import numpy as np


adc_dict = {
    'IN 0': 'patch',
    'IN 2': 'led',
    'IN 3': 'electrode',
    'IN 4': 'pmt',
    'IN 5': 'scanner',
    }


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


def get_pmt_gate(
        abf_path,
        bnc_V_threshold: float = 4.5):

    abf = pyabf.ABF(abf_path)
    adc_names = abf.adcNames

    abf.setSweep(0, channel=adc_names.index('IN 5'))
    scanner_adc_binary = np.where(abf.sweepY > bnc_V_threshold, 1, 0)
    indices_where_one = np.where(scanner_adc_binary == 1)[0]

    group_boundaries = np.where(np.diff(indices_where_one) > 1)[0] + 1
    scanner_on_indices = [list(group)
                          for group in np.split(
                                  indices_where_one, group_boundaries)]

    scanner_on_start_index = scanner_on_indices[0][0]  # Assume first scan
    scanner_on_start_timestamp = abf.sweepX[scanner_on_start_index]

    abf.setSweep(0, channel=adc_names.index('IN 4'))
    pmtgate_adc_binary = np.where(abf.sweepY > bnc_V_threshold, 1, 0)
    indices_where_one = np.where(pmtgate_adc_binary == 1)[0]

    pmtgate_on_start_index = indices_where_one[0]
    pmtgate_on_start_timestamp = abf.sweepX[pmtgate_on_start_index]

    pmtgate_on_end_index = indices_where_one[-1]
    pmtgate_on_end_timestamp = abf.sweepX[pmtgate_on_end_index]

    pmtgate_onset = pmtgate_on_start_timestamp - scanner_on_start_timestamp
    pmtgate_offset = pmtgate_on_end_timestamp - scanner_on_start_timestamp

    return pmtgate_onset, pmtgate_offset


def get_stim_time(
        abf_path,
        bnc_V_threshold: float = 4.5):

    abf = pyabf.ABF(abf_path)
    adc_names = abf.adcNames

    # possible_channels = ['IN 3', 'IN 2']
    # channel_name = next((ch for ch in possible_channels if ch in adc_names),
    #                     None)

    # if channel_name is None:
    #     raise ValueError()

    channel_name = 'IN 3'

    abf.setSweep(0, channel=adc_names.index(channel_name))

    stim_adc_binary = np.where(abf.sweepY > bnc_V_threshold, 1, 0)
    import matplotlib.pyplot as plt
    plt.plot(stim_adc_binary)
    indices_where_one = np.where(stim_adc_binary == 1)[0]

    group_boundaries = np.where(np.diff(indices_where_one) > 1)[0] + 1
    stim_on_indices = [list(group)
                       for group in np.split(
                               indices_where_one, group_boundaries)]

    stim_on_start_index = stim_on_indices[0][0]  # Assume first scan
    stim_on_start_timestamp = abf.sweepX[stim_on_start_index]

    abf.setSweep(0, channel=adc_names.index('IN 5'))
    scanner_adc_binary = np.where(abf.sweepY > bnc_V_threshold, 1, 0)
    import matplotlib.pyplot as plt
    plt.plot(scanner_adc_binary)
    indices_where_one = np.where(scanner_adc_binary == 1)[0]

    group_boundaries = np.where(np.diff(indices_where_one) > 1)[0] + 1
    scanner_on_indices = [list(group)
                          for group in np.split(
                                  indices_where_one, group_boundaries)]

    scanner_on_start_index = scanner_on_indices[1][0]  # Assume first scan
    scanner_on_start_timestamp = abf.sweepX[scanner_on_start_index]

    return float(round(
        stim_on_start_timestamp - scanner_on_start_timestamp))
