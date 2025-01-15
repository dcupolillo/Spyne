""" Created on Wed Jun  5 10:12:46 2024
    @author: dcupolillo """

# import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from spyne.core.utils.filters import modified_okada_filter


def rolling_quantile(
        array: tf.Tensor,
        window: int,
        min_quantile: float
) -> tf.Tensor:
    """
    Calculate the specified quantile for the given array.

    Parameters
    ----------
    array : tf.Tensor
        The input array for which the quantile needs to be calculated.
    window : int
        The size of the window for the rolling operation. This parameter is
        present but not directly used in this function.
    min_quantile : float
        The quantile to compute. For example, a `min_quantile` of 50 would
        compute the median.

    Returns
    -------
    tf.Tensor
        The calculated quantile for the input array.
    """

    # Cast to a float32
    array = tf.cast(array, tf.float32)

    # Compute the specified percentile using TensorFlow Probability
    q = tfp.stats.percentile(array, min_quantile, interpolation='linear')

    return q


def dFF(
        spine: object,
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

    # Initialize f as a TensorFlow empty variable
    # tf.Variable value can be changed at each iteration of the loop.
    f = tf.Variable(tf.zeros(spine.n_frames, dtype=tf.float32))

    # Update f in the loop
    for n, frame in enumerate(spine.roi[sweep_index].sweep[:, 0, :, :]):

        # Compute the minimum value of the frame using TensorFlow
        frame_min = tf.reduce_min(frame)

        with tf.device('/cpu:0'):
            # Subtract the minimum value from the frame to rescale it
            frame_rescaled = frame - frame_min

        # TensorFlow operations are optimized for tf.float32
        # frame_rescaled = frame_rescaled.astype(np.uint16)
        frame_rescaled = tf.cast(frame_rescaled, tf.float32)

        # Find the indices where the mask is greater than 0
        mask_indices = tf.where(spine.mask > 0)

        # Compute the mean value of the rescaled frame at the mask indices
        mean_value = tf.reduce_mean(tf.gather_nd(frame_rescaled, mask_indices))

        # Assign the computed mean value to the f tensor
        f[n].assign(mean_value)

    # Centered rolling quantile
    window = int(window_sec * spine.frame_rate)
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


# def get_time_series(
#         spine: object,
#         sweep_index: int,
# ) -> Union[np.ndarray, cp.ndarray]:
#     """
#     Generate a time series array for a given spine and sweep index.

#     Parameters
#     ----------
#     spine : object
#         The spine object containing the data.
#     sweep_index : int
#         The index of the sweep.
#     use_gpu : bool
#         Whether to use GPU acceleration with cupy.

#     Returns
#     -------
#     time_series : np.ndarray or cp.ndarray
#         The generated time series array, with each element representing
#         the time point corresponding to a frame index.
#     """

#     return np.array([frame_index / spine.frame_rate
#                      for frame_index in range(spine.n_frames)])

def get_time_series(
        spine: object,
        sweep_index: int,
) -> tf.Tensor:
    """
    Generate a time series array for a given spine and sweep index.

    Parameters
    ----------
    spine : object
        The spine object containing the data.
    sweep_index : int
        The index of the sweep.

    Returns
    -------
    time_series : tf.Tensor
        The generated time series array, with each element representing
        the time point corresponding to a frame index.
    """

    return tf.cast(
        tf.range(spine.n_frames, dtype=tf.float32) /
        spine.frame_rate, tf.float32)


# def z_score(
#         spine: object,
#         sweep_index: int,
# ) -> Union[np.ndarray, cp.ndarray]:
#     """
#     Calculate the z-score of the dFF trace for a given sweep index.

#     Parameters
#     ----------
#     sweep_index : int
#         The index of the sweep.
#     use_gpu : bool, optional
#         Whether to use GPU acceleration with cupy. Default is False.

#     Returns
#     -------
#     z_scores : np.ndarray or cp.ndarray
#         The calculated z-scores.
#     """

#     dff = np.array(spine.f(sweep_index))
#     mean = np.mean(dff)
#     st_dev = np.std(dff)

#     return (dff - mean) / st_dev

def z_score(
        spine: object,
        sweep_index: int,
) -> tf.Tensor:
    """
    Calculate the z-score of the dFF trace for a given sweep index.

    Parameters
    ----------
    sweep_index : int
        The index of the sweep.

    Returns
    -------
    z_scores : tf.Tensor
        The calculated z-scores.
    """

    dff = tf.convert_to_tensor(spine.f(sweep_index), dtype=tf.float32)
    mean = tf.reduce_mean(dff)
    st_dev = tf.math.reduce_std(dff)

    return tf.divide(tf.subtract(dff, mean), st_dev)
