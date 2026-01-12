""" Created on Wed Jun  5 10:12:46 2024
    @author: dcupolillo """

import numpy as np
import torch
from spyne.core.spines.analysis.timeseries.filters import modified_okada_filter


def dFF(
    n_frames: int,
    roi: np.ndarray,
    mask: np.ndarray,
    frame_rate: float,
    sweep_index: int,
    rolling_bsl: str,
    window_sec: float,
    min_quantile: int,
) -> torch.Tensor:
    """
    Calculate dFF for a given spine and sweep index.

    Parameters
    ----------
    spine : object
        The spine object containing the data.
    sweep_index : int
        The index of the sweep.
    rolling_bsl : str
        Type of rolling baseline to use ('centered' or other).
    window_sec : float
        Window size in seconds for rolling baseline.
    min_quantile : int
        Minimum quantile for baseline calculation.
    use_gpu : bool, optional
        Whether to use GPU acceleration with cupy. Default is False.

    Returns
    -------
    dff : tf.Tensor
        The calculated dFF trace.
    """

    sweep = roi[sweep_index].sweep[:, 0, :, :]  # Extract the sweep data
    sweep = torch.tensor(sweep, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    frame_rate = float(frame_rate)

    f = torch.zeros(n_frames, dtype=torch.float32)
    
    for n, frame in enumerate(sweep):
        
        frame_min = torch.min(frame)
        frame_rescaled = frame - frame_min
        frame_rescaled = frame_rescaled.float()
        mask_indices = torch.nonzero(mask > 0, as_tuple=False)
        mean_value = torch.mean(frame_rescaled[mask_indices[:, 0], mask_indices[:, 1]])
        
        f[n] = mean_value

    window = int(window_sec * frame_rate)
    half_window = int(window / 2)

    if rolling_bsl == 'centered':
        baseline = torch.tensor([
            torch.quantile(f[t - half_window: t + half_window + 1], min_quantile / 100.0)
            for t in range(half_window, len(f) - half_window)
        ], dtype=torch.float32)

        left_edge_baseline = torch.tensor([
            torch.quantile(f[: t + half_window + 1], min_quantile / 100.0)
            for t in range(half_window)
        ], dtype=torch.float32)

        right_edge_baseline = torch.tensor([
            torch.quantile(f[t - half_window:], min_quantile / 100.0)
            for t in range(len(f) - half_window, len(f))
        ], dtype=torch.float32)

        bl = torch.cat([left_edge_baseline, baseline, right_edge_baseline], dim=0)
    else:
        baseline = torch.tensor([
            torch.quantile(f[t: t + window], min_quantile / 100.0)
            for t in range(1, len(f) - window)
        ], dtype=torch.float32)

        missing = torch.quantile(f[-window:], min_quantile / 100.0)
        missing = missing.repeat(window + 1)
        bl = torch.cat([baseline, missing], dim=0)

    epsilon = 1e-10
    dff = (f - bl) / (bl + epsilon)
    dff = modified_okada_filter(dff)
    
    return dff


def get_timestamps(
    n_frames: int,
    frame_rate: float,
) -> torch.Tensor:
    """
    Generate a time series array for given frame count and frame rate.

    Parameters
    ----------
    n_frames : int
        Number of frames in the time series.
    frame_rate : float
        Frame rate in Hz.

    Returns
    -------
    time_series : tf.Tensor
        The generated time series array, with each element representing
        the time point corresponding to a frame index.
    """

    return torch.arange(n_frames, dtype=torch.float32) / frame_rate


def z_score(
    n_frames: int,
    roi: np.ndarray,
    mask: np.ndarray,
    frame_rate: float,
    sweep_index: int,
    rolling_bsl: str = 'centered',
    window_sec: float = 0.5,
    min_quantile: int = 10,
) -> torch.Tensor:
    """
    Calculate the z-score of the dFF trace for a given sweep index.

    Parameters
    ----------
    n_frames : int
        Number of frames in the time series.
    roi : np.ndarray
        ROI data containing sweep information.
    mask : np.ndarray
        Spine mask for extracting signal.
    frame_rate : float
        Frame rate in Hz.
    sweep_index : int
        The index of the sweep.
    rolling_bsl : str, optional
        Type of rolling baseline to use. Default is 'centered'.
    window_sec : float, optional
        Window size in seconds for rolling baseline. Default is 0.5.
    min_quantile : int, optional
        Minimum quantile for baseline calculation. Default is 10.

    Returns
    -------
    z_scores : tf.Tensor
        The calculated z-scores.
    """

    dff = dFF(
        n_frames=n_frames,
        roi=roi,
        mask=mask,
        frame_rate=frame_rate,
        sweep_index=sweep_index,
        rolling_bsl=rolling_bsl,
        window_sec=window_sec,
        min_quantile=min_quantile
    )
    mean = torch.mean(dff)
    st_dev = torch.std(dff)
    return (dff - mean) / (st_dev + 1e-10)
