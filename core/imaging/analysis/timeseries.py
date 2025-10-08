""" Created on Wed Jun  5 10:12:46 2024
    @author: dcupolillo """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from spyne.core.imaging.preprocessing import modified_okada_filter


def dFF(
        n_frames: int,
        roi: np.ndarray,
        mask: np.ndarray,
        frame_rate: float,
        sweep_index: int,
        rolling_bsl: str,
        window_sec: float,
        min_quantile: int,
) -> tf.Tensor:
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

    n_frames = tf.constant(n_frames, dtype=tf.int32)
    sweep = roi[sweep_index].sweep[:, 0, :, :]  # Extract the sweep data
    sweep = tf.convert_to_tensor(sweep, dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    frame_rate = tf.constant(frame_rate, dtype=tf.float32)

    # Initialize f as a TensorFlow empty variable
    # tf.Variable value can be changed at each iteration of the loop.
    f = tf.Variable(tf.zeros(n_frames, dtype=tf.float32))

    # Update f in the loop
    for n, frame in enumerate(sweep):

        # Compute the minimum value of the frame using TensorFlow
        frame_min = tf.reduce_min(frame)

        with tf.device('/cpu:0'):
            # Subtract the minimum value from the frame to rescale it
            frame_rescaled = frame - frame_min

        # TensorFlow operations are optimized for tf.float32
        # frame_rescaled = frame_rescaled.astype(np.uint16)
        frame_rescaled = tf.cast(frame_rescaled, tf.float32)

        # Find the indices where the mask is greater than 0
        mask_indices = tf.where(mask > 0)

        # Compute the mean value of the rescaled frame at the mask indices
        mean_value = tf.reduce_mean(tf.gather_nd(frame_rescaled, mask_indices))

        # Assign the computed mean value to the f tensor
        f[n].assign(mean_value)

    # Centered rolling quantile
    window = int(window_sec * frame_rate)
    half_window = int(window / 2)

    if rolling_bsl == 'centered':

        f_tensor = tf.convert_to_tensor(f)
        baseline = tf.convert_to_tensor(
            [tfp.stats.percentile(
                f_tensor[t - half_window: t + half_window + 1],
                min_quantile)
                for t in range(half_window, len(f_tensor) - half_window)],
            dtype=tf.float32)

        left_edge_baseline = tf.convert_to_tensor(
            [tfp.stats.percentile(
                f_tensor[: t + half_window + 1], min_quantile)
             for t in range(half_window)], dtype=tf.float32)

        right_edge_baseline = tf.convert_to_tensor(
            [tfp.stats.percentile(
                f_tensor[t - half_window:], min_quantile)
             for t in range(len(f_tensor) - half_window, len(f_tensor))],
            dtype=tf.float32)

        bl = tf.concat([
            left_edge_baseline, baseline, right_edge_baseline], axis=0)

    else:

        f_tensor = tf.convert_to_tensor(f)
        baseline = tf.convert_to_tensor(
            [tfp.stats.percentile(
                f_tensor[t: t + window], min_quantile)
             for t in range(1, len(f_tensor) - window)],
            dtype=tf.float32)

        missing = tfp.stats.percentile(f_tensor[-window:], min_quantile)
        missing = tf.repeat(missing, window + 1)
        bl = tf.concat([baseline, missing], axis=0)

    epsilon = 1e-10  # Ensure no division by zero
    dff = tf.divide(tf.subtract(f, bl), bl + epsilon)

    # Apply your custom filter (ensure it's converted to TensorFlow operations)
    dff = modified_okada_filter(dff)

    return dff


def get_timestamps(
        n_frames: int,
        frame_rate: float,
) -> tf.Tensor:
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

    return tf.cast(
        tf.range(n_frames, dtype=tf.float32) / frame_rate, tf.float32)


def z_score(
        n_frames: int,
        roi: np.ndarray,
        mask: np.ndarray,
        frame_rate: float,
        sweep_index: int,
        rolling_bsl: str = 'centered',
        window_sec: float = 0.5,
        min_quantile: int = 10,
) -> tf.Tensor:
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
    mean = tf.reduce_mean(dff)
    st_dev = tf.math.reduce_std(dff)

    return tf.divide(tf.subtract(dff, mean), st_dev)
