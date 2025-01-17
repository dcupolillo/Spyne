from tqdm import tqdm
from pathlib import Path
import numpy as np
import flammkuchen as fl
import tensorflow as tf


def collect_timeseries(
    segmenters: list,
    spines_data: list,
    device: str,
    output_folder: str or Path,
) -> tuple:
    """
    Collect and process timeseries data for all spines.

    Parameters
    ----------
    segmenters : list
        List of RoiSegmenter objects, one for each ROI in the dataset.
    spines_data : list
        Processed spine data from semantic segmentation.
    device : str
        Device to perform calculations ('/GPU:0' or '/CPU:0').
    output_folder : str or Path
        Folder to save the collected timeseries data.

    Returns
    -------
    tuple
        Processed timeseries data:
        - `z_scores_CA3` : np.ndarray
        - `dFF_CA3` : np.ndarray
        - `ts_CA3` : np.ndarray
        - `z_scores_BLA` : np.ndarray
        - `dFF_BLA` : np.ndarray
        - `ts_BLA` : np.ndarray
    """
    output_folder = Path(output_folder)
    total_spines = len(spines_data)

    # Initialize lists
    zscores_CA3 = []
    dFF_CA3 = []
    ts_CA3 = []

    zscores_BLA = []
    dFF_BLA = []
    ts_BLA = []

    with tf.device(device):

        with tqdm(total=total_spines, desc="Collecting timeseries") as pbar:

            for roi_segmenter in segmenters:

                roi_n = roi_segmenter.n_roi

                n_spines = len(
                    [z for z in spines_data
                     if z['roi_n'] == roi_n])

                if n_spines == 0:
                    continue

                n_IN3 = roi_segmenter.adc_list.count('IN 3')
                n_IN2 = roi_segmenter.adc_list.count('IN 2')

                for spine_index, spine in enumerate(roi_segmenter):

                    z_scores_CA3 = np.empty(n_IN3, dtype=object)
                    dff_values_CA3 = np.empty(n_IN3, dtype=object)
                    timestamps_CA3 = np.empty(n_IN3, dtype=object)
                    z_scores_BLA = np.empty(n_IN2, dtype=object)
                    dff_values_BLA = np.empty(n_IN2, dtype=object)
                    timestamps_BLA = np.empty(n_IN2, dtype=object)

                    index_CA3 = 0
                    index_BLA = 0

                    for sweep_index in range(roi_segmenter.n_sweeps):

                        adc_value = roi_segmenter.adc_list[sweep_index]

                        z_score = spine.zscore(sweep_index)
                        dff_value = spine.f(sweep_index)
                        timestamp = spine.ft(sweep_index)

                        if adc_value == 'IN 3':
                            z_scores_CA3[index_CA3] = z_score
                            dff_values_CA3[index_CA3] = dff_value
                            timestamps_CA3[index_CA3] = timestamp
                            index_CA3 += 1

                        elif adc_value == 'IN 2':
                            z_scores_BLA[index_BLA] = z_score
                            dff_values_BLA[index_BLA] = dff_value
                            timestamps_BLA[index_BLA] = timestamp
                            index_BLA += 1
                        else:
                            print("unknown adc")

                    # Sanity check to ensure sweep number consistency
                    assert (index_CA3 + index_BLA) == (sweep_index + 1)

                    zscores_CA3.append(list(z_scores_CA3))
                    dFF_CA3.append(list(dff_values_CA3))
                    ts_CA3.append(list(timestamps_CA3))
                    zscores_BLA.append(list(z_scores_BLA))
                    dFF_BLA.append(list(dff_values_BLA))
                    ts_BLA.append(list(timestamps_BLA))

                    pbar.update(1)

                if not n_spines == 0:
                    assert spine_index == (n_spines - 1)

    # Save the arrays to disk
    fl.save(output_folder / "zscores_CA3.h5", zscores_CA3)
    fl.save(output_folder / "dFF_CA3.h5", dFF_CA3)
    fl.save(output_folder / "ts_CA3.h5", ts_CA3)
    fl.save(output_folder / "zscores_BLA.h5", zscores_BLA)
    fl.save(output_folder / "dFF_BLA.h5", dFF_BLA)
    fl.save(output_folder / "ts_BLA.h5", ts_BLA)

    return zscores_CA3, dFF_CA3, ts_CA3, zscores_BLA, dFF_BLA, ts_BLA
