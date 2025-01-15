""" Created on Mon Oct 30 12:27:04 2023
    @author: dcupolillo """

import numpy as np
import tensorflow as tf
import cv2
from typing import Union, Tuple


def median(image, kernel: int):
    """
    Image filtering.

    Parameters
    ----------
    image : np.ndarray
        The input image to be filtered.
    kernel : int
        DESCRIPTION.

    Raises
    ------
    ValueError
        Kernel size must be odd.

    Returns
    -------
    np.ndarray
        Filtered image.

    """

    if kernel % 2 == 0:
        raise ValueError("Kernel size must be odd")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return cv2.medianBlur(image, kernel)


def gaussian(image, kernel: int, sigma: float) -> np.ndarray:
    """
    Gaussian filter implementation.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    kernel : int
        Kernel size.
    sigma : float
        Gaussian kernel standard deviation.

    Returns
    -------
    np.ndarray
        The blurred image.

    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return cv2.GaussianBlur(image, (kernel, kernel), sigma)


# def modified_okada_filter(
#         time_series: Union[np.ndarray, cp.ndarray],
# ) -> Union[np.ndarray, cp.ndarray]:
#     """
#     Trace filtering.
#     Ishikawa et al.,
#     "Functional Multiple-Spine Calcium Imaging from Brain Slices"
#     STAR Protocols (2020), https://doi.org/10.1016/j.xpro.2020.100121
#     Figure 8

#     Parameters
#     ----------
#     time_series : np.ndarray or cp.ndarray
#         The input dFF trace.
#     use_gpu : bool
#         Whether to use GPU acceleration with cupy.

#     Returns
#     -------
#     filtered_series : np.ndarray or cp.ndarray
#         Filtered dFF trace.
#     """

#     filtered_series = np.copy(time_series)
#     n_points = len(time_series)

#     # Calculate mean and standard deviation of the entire time series
#     mean = np.mean(time_series)
#     st_dev = np.std(time_series)

#     for t in range(1, n_points - 1):

#         xt = time_series[t]
#         xt_minus_1 = time_series[t - 1]
#         xt_plus_1 = time_series[t + 1]

#         # Z is defined as signal saliency against background noise
#         Z = np.abs((xt_plus_1 - mean) / st_dev)

#         # Check if xt is the median
#         if (xt - xt_minus_1) * (xt - xt_plus_1) > 0:
#             filtered_series[t] = (xt_minus_1 + Z * xt + xt_plus_1) / (2 + Z)

#     return filtered_series

def condition(
        t: tf.Tensor,
        filtered_series: tf.Tensor,
        time_series: tf.Tensor,
        mean: tf.Tensor,
        st_dev: tf.Tensor,
        n_points: tf.Tensor
) -> tf.Tensor:
    """
    Condition for the TensorFlow while loop.

    Parameters
    ----------
    t : tf.Tensor
        The current time point in the loop.
    filtered_series : tf.Tensor
        The filtered time series.
    time_series : tf.Tensor
        The original time series.
    mean : tf.Tensor
        The mean of the time series.
    st_dev : tf.Tensor
        The standard deviation of the time series.
    n_points : tf.Tensor
        The number of points in the time series.

    Returns
    -------
    tf.Tensor
        A boolean tensor indicating whether to continue the loop.
    """
    return tf.less(t, n_points - 1)


def body(
        t: tf.Tensor,
        filtered_series: tf.Tensor,
        time_series: tf.Tensor,
        mean: tf.Tensor, st_dev: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Body of the TensorFlow while loop.

    Parameters
    ----------
    t : tf.Tensor
        The current time point in the loop.
    filtered_series : tf.Tensor
        The filtered time series.
    time_series : tf.Tensor
        The original time series.
    mean : tf.Tensor
        The mean of the time series.
    st_dev : tf.Tensor
        The standard deviation of the time series.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        The updated time point and filtered series.
    """
    xt = time_series[t]
    xt_minus_1 = time_series[t - 1]
    xt_plus_1 = time_series[t + 1]

    # Z is defined as signal saliency against background noise
    Z = tf.abs((xt_plus_1 - mean) / st_dev)

    # Check if xt is the median
    condition = (xt - xt_minus_1) * (xt - xt_plus_1) > 0
    updated_xt = (xt_minus_1 + Z * xt + xt_plus_1) / (2 + Z)

    # Update filtered_series at index t
    updated_xt = tf.reshape(tf.where(condition, updated_xt, xt), [1])
    filtered_series = tf.tensor_scatter_nd_update(
        filtered_series, [[t]], tf.where(condition, updated_xt, xt))

    return t + 1, filtered_series, time_series, mean, st_dev


def modified_okada_filter(
        time_series: tf.Tensor
) -> tf.Tensor:
    """
    Trace filtering.
    Ishikawa et al.,
    "Functional Multiple-Spine Calcium Imaging from Brain Slices"
    STAR Protocols (2020), https://doi.org/10.1016/j.xpro.2020.100121
    Figure 8

    Parameters
    ----------
    time_series : tf.Tensor
        The input dFF trace.

    Returns
    -------
    filtered_series : tf.Tensor
        Filtered dFF trace.
    """
    filtered_series = tf.identity(time_series)  # Make a copy
    n_points = tf.shape(time_series)[0]

    # Calculate mean and standard deviation of the entire time series
    mean = tf.reduce_mean(time_series)
    st_dev = tf.math.reduce_std(time_series)

    # Execute the while loop starting from t=1
    t = tf.constant(1)
    loop_vars = [t, filtered_series, time_series, mean, st_dev]
    t, filtered_series, _, _, _ = tf.while_loop(
        lambda t, filtered_series, time_series, mean, st_dev:
            condition(t, filtered_series, time_series, mean, st_dev, n_points),
        lambda t, filtered_series, time_series, mean, st_dev:
            body(t, filtered_series, time_series, mean, st_dev),
        loop_vars
    )

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
