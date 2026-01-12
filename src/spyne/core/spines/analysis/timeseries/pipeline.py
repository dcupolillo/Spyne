from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

from spyne.core.spines.analysis.timeseries.timeseries import dFF, get_timestamps, z_score


def collect_timeseries(
    dataset,
    spines_data: list,
    metadata: dict,
    device: str,
    output_folder: str or Path,
) -> tuple:
    """
    Collect and process timeseries data for all spines.

    Parameters
    ----------
    dataset : ImagingDataset
        The imaging dataset containing ROI data.
    spines_data : list
        Processed spine data from semantic segmentation.
    metadata : dict
        ROI metadata dictionary containing frame rates, ADC lists, etc.
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

    if total_spines == 0:
        # Return empty arrays if no spines
        empty_array = np.array([])
        return (
            empty_array,
            empty_array,
            empty_array,
            empty_array,
            empty_array,
            empty_array)

    # Get metadata from first ROI to determine array shapes
    first_roi_meta = metadata[0]
    n_frames = first_roi_meta['n_frames']
    n_IN3 = first_roi_meta['adc_list'].count('IN 3')
    n_IN2 = first_roi_meta['adc_list'].count('IN 2')

    # Pre-allocate numpy arrays with shape (n_spines, n_sweeps, n_frames)
    zscores_CA3 = np.empty((total_spines, n_IN3, n_frames), dtype=np.float32)
    dFF_CA3 = np.empty((total_spines, n_IN3, n_frames), dtype=np.float32)
    ts_CA3 = np.empty((total_spines, n_IN3, n_frames), dtype=np.float32)

    zscores_BLA = np.empty((total_spines, n_IN2, n_frames), dtype=np.float32)
    dFF_BLA = np.empty((total_spines, n_IN2, n_frames), dtype=np.float32)
    ts_BLA = np.empty((total_spines, n_IN2, n_frames), dtype=np.float32)

    # Fill arrays with NaN for missing data
    zscores_CA3.fill(np.nan)
    dFF_CA3.fill(np.nan)
    ts_CA3.fill(np.nan)
    zscores_BLA.fill(np.nan)
    dFF_BLA.fill(np.nan)
    ts_BLA.fill(np.nan)


    with tqdm(total=total_spines, desc="Collecting timeseries") as pbar:
        for global_spine_index, spine_data in enumerate(spines_data):
            roi_n = spine_data['roi_n']
            roi_meta = metadata[roi_n]
            roi_data = dataset[roi_n]

            spine_mask = spine_data['mask']
            n_frames = roi_meta['n_frames']
            frame_rate = roi_meta['frame_rate']

            index_CA3 = 0
            index_BLA = 0

            for sweep_index in range(roi_meta['n_sweeps']):
                adc_value = roi_meta['adc_list'][sweep_index]

                # Calculate timeseries using core functions directly
                z_tensor = z_score(
                    n_frames=n_frames,
                    roi=roi_data,
                    mask=spine_mask,
                    frame_rate=frame_rate,
                    sweep_index=sweep_index
                )
                dff_tensor = dFF(
                    n_frames=n_frames,
                    roi=roi_data,
                    mask=spine_mask,
                    frame_rate=frame_rate,
                    sweep_index=sweep_index,
                    rolling_bsl='centered',
                    window_sec=0.5,
                    min_quantile=10
                )
                ts_tensor = get_timestamps(
                    n_frames=n_frames,
                    frame_rate=frame_rate
                )

                # Move tensors to device if needed
                if device == 'cuda' or device == 'cuda:0':
                    z_tensor = z_tensor.to('cuda')
                    dff_tensor = dff_tensor.to('cuda')
                    ts_tensor = ts_tensor.to('cuda')

                # Convert to numpy
                z_score_np = z_tensor.cpu().numpy()
                dff_value_np = dff_tensor.cpu().numpy()
                timestamp_np = ts_tensor.cpu().numpy()

                if adc_value == 'IN 3':
                    zscores_CA3[global_spine_index, index_CA3, :] = z_score_np
                    dFF_CA3[global_spine_index, index_CA3, :] = dff_value_np
                    ts_CA3[global_spine_index, index_CA3, :] = timestamp_np
                    index_CA3 += 1

                elif adc_value == 'IN 2':
                    zscores_BLA[global_spine_index, index_BLA, :] = z_score_np
                    dFF_BLA[global_spine_index, index_BLA, :] = dff_value_np
                    ts_BLA[global_spine_index, index_BLA, :] = timestamp_np
                    index_BLA += 1
                else:
                    print("unknown adc")

            assert (index_CA3 + index_BLA) == roi_meta['n_sweeps']
            pbar.update(1)

    assert zscores_CA3.shape == dFF_CA3.shape == ts_CA3.shape, "Inconsistent shapes for CA3 timeseries data"
    assert zscores_BLA.shape == dFF_BLA.shape == ts_BLA.shape, "Inconsistent shapes for BLA timeseries data"

    arrays = {i: array for i, array in enumerate(
        ['zscores_CA3', 'dFF_CA3', 'ts_CA3', 'zscores_BLA', 'dFF_BLA', 'ts_BLA'])}
    
    for name, array in arrays.items():
        if array.shape[0] != total_spines:
            raise ValueError(f"Mismatch in number of spines and {name} shape")

    return zscores_CA3, dFF_CA3, ts_CA3, zscores_BLA, dFF_BLA, ts_BLA
