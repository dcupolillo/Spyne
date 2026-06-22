""" Created on Wed Jun  5 10:12:46 2024
    @author: dcupolillo """

from __future__ import annotations
import numpy as np
import torch
from skimage.morphology import dilation, disk, remove_small_objects, erosion
from spyne.core.spines.analysis.timeseries.filters import modified_okada_filter


def _shift_to_zero(frame: torch.Tensor) -> torch.Tensor:
    """
    Shift frame pixel values so the minimum is zero.

    Parameters
    ----------
    frame : torch.Tensor
        2D tensor of pixel intensities for a single frame.

    Returns
    -------
    torch.Tensor
        Frame with minimum value subtracted, cast to float32.
    """
    return (frame - torch.min(frame)).float()


def background(
    n_frames: int,
    roi: np.ndarray,
    spine_mask: np.ndarray,
    dendrite_mask: np.ndarray,
    sweep_index: int,
    dilation_disk_size: int,
) -> torch.Tensor:
    """
    Calculate the background fluorescence signal for a given roi and sweep index.
    Background area is defined as the area outside the dilated spine and dendrite masks.
    Noise is calculated as the median of the pixel values in the background area for each frame.
    
    Parameters
    ----------
    n_frames : int
        Number of frames in the time series.
    roi : np.ndarray
        ROI data containing sweep information.
    spine_mask : np.ndarray
        Spine mask for extracting signal.
    dendrite_mask : np.ndarray
        Dendrite mask for extracting signal.
    sweep_index : int
        The index of the sweep to process.
    dilation_disk_size : int
        The size of the disk used for dilation.
    
    Returns
    -------
    background : torch.Tensor
        The calculated background fluorescence signal.
    """
    
    sweep = roi[sweep_index].sweep[:, 0, :, :]  # (n_frames, H, W)
    sweep = torch.tensor(sweep, dtype=torch.float32)

    # Combine spines and dendrite mask
    combined_mask = (spine_mask > 0) | (dendrite_mask > 0)

    # Dilate combined mask
    dilated_spines_dendrite_mask = dilation(
        combined_mask.astype(bool), disk(dilation_disk_size))

    # Create background mask by inverting the dilated mask
    background_mask = torch.tensor(
        ~dilated_spines_dendrite_mask, dtype=torch.bool)

    if not background_mask.any():
        # Dilated combined mask covers the entire frame;
        # return zeros so dFF falls back to no neuropil correction.
        return torch.zeros(n_frames, dtype=torch.float32)

    # Vectorised: shift each frame to zero-minimum, then take median over
    # background pixels across all frames at once.
    # sweep: (n_frames, H, W) -> pixels: (n_frames, n_bg_pixels)
    sweep_min = sweep.amin(dim=(1, 2), keepdim=True)
    sweep_shifted = sweep - sweep_min
    bg_pixels = sweep_shifted[:, background_mask]  # (n_frames, n_bg_pixels)
    return bg_pixels.median(dim=1).values



def dFF(
    n_frames: int,
    roi: np.ndarray,
    mask: np.ndarray,
    frame_rate: float,
    sweep_index: int,
    rolling_bsl: str,
    window_sec: float,
    min_quantile: int,
    background_signal: torch.Tensor,
    neuropil_factor: float,
) -> torch.Tensor:
    """
    Calculate dFF for a given spine and sweep index.

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
    rolling_bsl : str
        Type of rolling baseline to use ('centered' or other).
    window_sec : float
        Window size in seconds for rolling baseline.
    min_quantile : int
        Minimum quantile for baseline calculation.
    background_signal : torch.Tensor
        Per-frame background fluorescence, precomputed once per ROI/sweep
        using :func:`background`. Scaled by ``neuropil_factor`` before
        subtraction to avoid collapsing the resting baseline to zero.
    neuropil_factor : float, optional
        Fraction of background signal to subtract (neuropil contamination
        coefficient). 0.7 is the commonly used default (suite2p / Chen 2013).
        Must be in [0, 1]. Default is 0.7.

    Returns
    -------
    dff : torch.Tensor
        The calculated dFF trace.
    """

    sweep = roi[sweep_index].sweep[:, 0, :, :]  # (n_frames, H, W)
    sweep = torch.tensor(sweep, dtype=torch.float32)
    mask_bool = torch.tensor(mask > 0, dtype=torch.bool)
    frame_rate = float(frame_rate)

    # Vectorised: shift each frame to zero-minimum, then mean over mask pixels.
    sweep_min = sweep.amin(dim=(1, 2), keepdim=True)
    sweep_shifted = sweep - sweep_min
    spine_pixels = sweep_shifted[:, mask_bool]  # (n_frames, n_spine_pixels)
    f = spine_pixels.mean(dim=1)               # (n_frames,)

    # Neuropil correction (suite2p approach):
    #   F_corr = F - r·Fneu
    # The rolling quantile baseline computed below absorbs any DC shift,
    # so no need to add back mean(Fneu).
    f = f - (neuropil_factor * background_signal)

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
    dff_signal: torch.Tensor | None = None,
    baseline_percentile: float = 25.0,
    n_frames: int | None = None,
    roi: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    frame_rate: float | None = None,
    sweep_index: int | None = None,
    rolling_bsl: str = 'centered',
    window_sec: float = 0.5,
    min_quantile: int = 10,
    background_signal: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculate the z-score of a dFF trace using a robust baseline.

    Mean and standard deviation are estimated only from frames at or
    below ``baseline_percentile``, so large calcium transients do not
    inflate the noise estimate.

    Accepts either a precomputed ``dff_signal`` (preferred, avoids
    redundant computation) or the raw parameters needed to compute
    dFF internally.

    Parameters
    ----------
    dff_signal : torch.Tensor, optional
        Precomputed dFF trace. If provided, raw parameters are ignored.
    baseline_percentile : float, optional
        Percentile threshold (0–100) below which frames are considered
        baseline. Default is 25.0.
    n_frames : int, optional
        Number of frames in the time series.
    roi : np.ndarray, optional
        ROI data containing sweep information.
    mask : np.ndarray, optional
        Spine mask for extracting signal.
    frame_rate : float, optional
        Frame rate in Hz.
    sweep_index : int, optional
        The index of the sweep.
    rolling_bsl : str, optional
        Type of rolling baseline to use. Default is 'centered'.
    window_sec : float, optional
        Window size in seconds for rolling baseline. Default is 0.5.
    min_quantile : int, optional
        Minimum quantile for baseline calculation. Default is 10.
    background_signal : torch.Tensor, optional
        Per-frame background fluorescence, precomputed once per ROI/sweep
        using :func:`background`. Passed through to :func:`dFF` when
        ``dff_signal`` is not provided. Default is None.

    Returns
    -------
    z_scores : torch.Tensor
        The calculated z-scores.
    """
    if dff_signal is None:
        if any(v is None for v in (n_frames, roi, mask, frame_rate, sweep_index)):
            raise ValueError(
                "Provide either 'dff_signal' or all of: "
                "n_frames, roi, mask, frame_rate, sweep_index."
            )
        dff_signal = dFF(
            n_frames=n_frames,
            roi=roi,
            mask=mask,
            frame_rate=frame_rate,
            sweep_index=sweep_index,
            rolling_bsl=rolling_bsl,
            window_sec=window_sec,
            min_quantile=min_quantile,
            background_signal=background_signal,
        )
    threshold = torch.quantile(
        dff_signal, baseline_percentile / 100.0)
    baseline_frames = dff_signal[dff_signal <= threshold]
    mean = baseline_frames.mean()
    st_dev = baseline_frames.std()

    return (dff_signal - mean) / (st_dev + 1e-10)
