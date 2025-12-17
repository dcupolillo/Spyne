""" Created on Mon Oct 30 12:27:04 2023
    @author: dcupolillo """

import numpy as np
import torch


def modified_okada_filter(time_series: torch.Tensor) -> torch.Tensor:
    """
    Trace filtering (PyTorch version).
    Ishikawa et al.,
    "Functional Multiple-Spine Calcium Imaging from Brain Slices"
    STAR Protocols (2020), https://doi.org/10.1016/j.xpro.2020.100121
    Figure 8

    Parameters
    ----------
    time_series : torch.Tensor
        The input dFF trace.

    Returns
    -------
filtered_series : torch.Tensor
        Filtered dFF trace.
    """

    if isinstance(time_series, np.ndarray):
        time_series = torch.from_numpy(time_series).float()
    
    filtered_series = time_series.clone()
    n_points = time_series.shape[0]
    mean = torch.mean(time_series)
    st_dev = torch.std(time_series)

    for t in range(1, n_points):
        xt = time_series[t]
        xt_minus_1 = time_series[t - 1]
        xt_plus_1 = time_series[t + 1] if t + 1 < n_points else time_series[t]

        z_raw = (xt_plus_1 - mean) / (st_dev + 1e-10)
        Z = torch.abs(z_raw)

        is_extrema = ((xt - xt_minus_1) * (xt - xt_plus_1)) > 0
        xt_updated = (xt_minus_1 + Z * xt + xt_plus_1) / (2.0 + Z)
        xt_new = xt_updated if is_extrema else xt

        filtered_series[t] = xt_new
    
    return filtered_series


def ewma(time_series: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Exponential weighted moving average
    Hongbo Jia et al., (Konnerth A.)
    "In vivo two-photon imaging of sensory-evoked
    dendritic calcium signals in cortical neurons"
    Nature protocols (2011)
    Box 1

    Parameters
    ----------
    time_series : np.ndarray
        The input dFF trace.
    alpha : float, optional
        DESCRIPTION. The default is 0.2.

    Returns
    -------
    ewma : TYPE
        Filtered dFF trace.

    """
    n_points = len(time_series)
    ewma = np.empty_like(time_series, dtype=float)
    ewma[0] = time_series[0]
    for t in range(1, n_points):
        ewma[t] = alpha * time_series[t] + (1 - alpha) * ewma[t - 1]
    return ewma