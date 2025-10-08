""" Created on Tue Jul 15 12:21:27 2025
    @author: dcupolillo """

import numpy as np
from scipy.ndimage import uniform_filter, median_filter


def radius_to_kernel_size(
        radius_um: float,
        pixel_size_xy: tuple,
        ensure_odd: bool = True,
        min_size: int = 3
) -> tuple:
    """
    Convert radius in µm to kernel size in pixels.
    
    Parameters:
        radius_um: radius in micrometers
        pixel_size_xy: physical size of pixels in microns (x, y)
        ensure_odd: if True, ensures kernel size is odd (standard for filters)
        min_size: minimum kernel size in pixels (default 3 for effective filtering)
    
    Returns:
        Tuple of (kernel_height, kernel_width) in pixels
    """
    # Convert radius to diameter (kernel size)
    diameter_um = 2 * radius_um
    
    # Convert to pixels
    kernel_width = max(min_size, round(diameter_um / pixel_size_xy[0]))
    kernel_height = max(min_size, round(diameter_um / pixel_size_xy[1]))
    
    # Ensure odd kernel sizes if requested (standard for most filters)
    if ensure_odd:
        if kernel_width % 2 == 0:
            kernel_width += 1
        if kernel_height % 2 == 0:
            kernel_height += 1
    
    return (kernel_height, kernel_width)


def compute_local_contrast(
        reference_frame: np.ndarray,
        pixel_size_xy: tuple,
        desired_window_um: float = 2.0
) -> np.ndarray:
    """
    Compute normalized local contrast from the morphology (reference) frame,
    using a window size defined in microns.
    """
    window_size = (
        max(1, round(desired_window_um / pixel_size_xy[1])),  # height (y)
        max(1, round(desired_window_um / pixel_size_xy[0]))   # width (x)
    )

    local_mean = uniform_filter(reference_frame, size=window_size)
    local_sqr_mean = uniform_filter(reference_frame**2, size=window_size)
    local_std = np.sqrt(
        np.clip(local_sqr_mean - local_mean**2, a_min=0, a_max=None))
    norm_contrast = (local_std - np.min(local_std)) / np.ptp(local_std)

    return norm_contrast


def compute_local_noise(
        reference_video: np.ndarray,
        pixel_size_xy: tuple,
        desired_window_um: float = 2.0
) -> np.ndarray:
    """
    Estimate local noise as standard deviation across time, normalized,
    using a window size defined in microns.
    """
    window_size = (
        max(1, round(desired_window_um / pixel_size_xy[1])),  # height (y)
        max(1, round(desired_window_um / pixel_size_xy[0]))   # width (x)
    )
    temporal_std = np.std(reference_video, axis=0)
    local_mean = uniform_filter(temporal_std, size=window_size)
    local_sqr_mean = uniform_filter(temporal_std**2, size=window_size)
    local_noise = np.sqrt(
        np.clip(local_sqr_mean - local_mean**2, a_min=0, a_max=None))
    norm_noise = (local_noise - np.min(local_noise)) / np.ptp(local_noise)

    return norm_noise


def map_to_kernel(
        norm_contrast: np.ndarray,
        norm_noise: np.ndarray,
        pixel_size_xy: tuple,
        desired_kernel_um: float = 1.5,
        min_k: int = 3,
        max_k: int = 7,
        roi_shape: tuple = None,
        alpha: float = 0.5
) -> np.ndarray:
    """
    Map combined contrast and noise to adaptive kernel size in pixels.
    """
    # Convert desired kernel size from µm to pixels
    k_px_x = round(desired_kernel_um / pixel_size_xy[0])
    k_px_y = round(desired_kernel_um / pixel_size_xy[1])
    base_kernel = min(k_px_x, k_px_y)
    base_kernel = max(min_k, min(base_kernel, max_k))

    # Combine contrast and noise (both in [0, 1])
    combined_score = alpha * (1.0 - norm_contrast) + (1.0 - alpha) * norm_noise
    mapped_kernel = min_k + combined_score * (max_k - min_k)
    mapped_kernel = np.round(mapped_kernel).astype(int)
    mapped_kernel += 1 - (mapped_kernel % 2)

    if roi_shape:
        roi_h, roi_w = roi_shape
        max_allowed = min(roi_h, roi_w) // 3
        mapped_kernel = np.clip(
            mapped_kernel, min_k, max_allowed | 1)

    return mapped_kernel


def adaptive_3d_median_filter(
        calcium_video: np.ndarray,
        morphology_video: np.ndarray,
        pixel_size_xy: tuple,
        frame_rate_hz: float,
        desired_kernel_um: float = 1.5,
        desired_window_um: float = 2.0,
        desired_temporal_kernel: float = 0.0625,
        min_kernel: int = 3,
        max_kernel: int = 7,
        alpha: float = 0.5,
) -> np.ndarray:
    """
    Apply adaptive 3D median filtering to calcium video
    using morphology video for contrast and noise estimation.

    Parameters:
        calcium_video: (T, H, W) array to be filtered
        morphology_video: (T, H, W) array used for computing contrast and noise
        pixel_size_xy: physical size of pixels in microns (x, y)
        frame_rate_hz: acquisition rate in Hz (frames per second)
        desired_kernel_um: target physical size for spatial kernel
        desired_window_um: target window size for contrast/noise estimation
        desired_temporal_kernel: target duration for temporal kernel in seconds
        alpha: weight for contrast (vs noise) in kernel mapping
    Returns:
        Filtered calcium video (T, H, W)
    """
    T, H, W = calcium_video.shape
    avg_frame = np.mean(morphology_video, axis=0)
    norm_contrast = compute_local_contrast(
        avg_frame, pixel_size_xy, desired_window_um)
    norm_noise = compute_local_noise(
        morphology_video, pixel_size_xy, desired_window_um)

    kernel_map = map_to_kernel(
        norm_contrast,
        norm_noise,
        pixel_size_xy=pixel_size_xy,
        desired_kernel_um=desired_kernel_um,
        min_k=min_kernel,
        max_k=max_kernel,
        roi_shape=(H, W),
        alpha=alpha)

    k_t = max(1, round(desired_temporal_kernel * frame_rate_hz))
    if k_t % 2 == 0:
        k_t += 1  # ensure odd temporal size

    output = np.zeros_like(calcium_video, dtype=np.float32)
    mask_accum = np.zeros_like(calcium_video, dtype=np.float32)

    unique_kernels = np.unique(kernel_map)
    for k in unique_kernels:
        if k < 3:
            continue
        k3d = (k_t, k, k)
        filtered = median_filter(calcium_video, size=k3d, mode='reflect')

        mask2d = (kernel_map == k).astype(np.float32)
        mask3d = np.broadcast_to(mask2d[None, :, :], calcium_video.shape)

        output += filtered * mask3d
        mask_accum += mask3d

    mask_accum[mask_accum == 0] = 1

    return (output / mask_accum).astype(np.int16)
