""" Created on Tue Oct  3 15:20:10 2023
    @author: dcupolillo """

import math
import numpy as np
import random
import tensorflow as tf
from typing import List, Tuple, Union
import time
from datetime import timedelta
from scipy import stats


def generate_uuid(
) -> tuple:

    uuid_hex = ''.join(random.choices('0123456789ABCDEF', k=16))
    uuid_uint64 = int(uuid_hex, 16)
    uuid_str = "{:.9e}".format(uuid_uint64)

    return uuid_hex, uuid_str


def format_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        return f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        mins, secs = divmod(elapsed_time, 60)
        return f"{int(mins)} mins {secs:.2f} secs"
    else:
        hours, remainder = divmod(elapsed_time, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{int(hours)} hours {int(mins)} mins {secs:.2f} secs"


def array_to_tuple(arrays: np.ndarray) -> Tuple:
    """
    Convert a numpy array to a hashable tuple for caching.
    """
    return tuple(map(tuple, arrays.reshape(arrays.shape[0], -1)))


def arrays_to_tuple(
        arrays: Union[np.ndarray, tf.Tensor]
) -> Tuple:
    """Convert a numpy array or TensorFlow tensor to a tuple of tuples."""
    if isinstance(arrays, tf.Tensor):
        arrays = arrays.numpy()  # Convert TensorFlow tensor to NumPy array
    return tuple(array_to_tuple(array) for array in arrays)


def segmenter_to_hashable(segmenter: object) -> Tuple:
    """
    Convert an RoiSegmenter object to a hashable representation.

    Parameters
    ----------
    segmenter : RoiSegmenter
        The RoiSegmenter object to convert.

    Returns
    -------
    Tuple
        A tuple representing the hashable attributes of the segmenter.
    """
    return (
        segmenter.scanfield_index,
        segmenter.roi_index,
        segmenter.dataset
    )


def find_corners(
        rotation: float,
        size: list,
        center: list
) -> list:
    """
    Finds the xy coordinates of the corners of a rotated rectangle.

    Parameters
    ----------
    rotation : float
        DESCRIPTION.
    size : list
        DESCRIPTION.
    center : list
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    angle_rad = math.radians(rotation)
    half_height = size[1] / 2
    half_width = size[0] / 2
    x_center, y_center = center

    x_tl = x_center - half_width * \
        math.cos(angle_rad) + half_height * math.sin(angle_rad)
    y_tl = y_center - half_width * \
        math.sin(angle_rad) - half_height * math.cos(angle_rad)
    x_tr = x_center + half_width * \
        math.cos(angle_rad) + half_height * math.sin(angle_rad)
    y_tr = y_center + half_width * \
        math.sin(angle_rad) - half_height * math.cos(angle_rad)
    x_br = x_center - half_width * \
        math.cos(angle_rad) - half_height * math.sin(angle_rad)
    y_br = y_center - half_width * \
        math.sin(angle_rad) + half_height * math.cos(angle_rad)
    x_bl = x_center + half_width * \
        math.cos(angle_rad) - half_height * math.sin(angle_rad)
    y_bl = y_center + half_width * \
        math.sin(angle_rad) + half_height * math.cos(angle_rad)

    corners = [[x_tl, y_tl], [x_tr, y_tr],
               [x_bl, y_bl], [x_br, y_br]]

    return corners


def transform(
        point_to_transform,
        sf_to_ref_T: np.ndarray,
        pix_to_ref_T: np.ndarray,
        center_xy: list,
        pixelresolution_xy: list
) -> np.ndarray:
    """
    Transforms pixel coordinates directly to scanner space coordinates.

    This function accounts for different scanner (ResScan / LinScan)
    field-of-views (FOVs) by using a common normalized reference space.

    The reference space has X and Y coordinates ranging from 0 to 1,
    representing the maximum extents of the scanners
    while maintaining their true aspect ratios.

    The scanner space is mapped into the common reference space
    via affine transformation.
    All Scanfields (including RotatedRectangle) defined within scanner space
    are mapped to reference space via affine transformation.

    Scanfields of type RotatedRectangle have two associated
    affine matrices that allow coordinate space conversions:
        - pixelToRefTransfrom: transform pixel coordinates to reference space
        - affine: transform scanfield coordinates to reference space.

    The inverted matrix T^-1 allows for the opposite transformation.

    Parameters
    ----------
    point_to_transform : TYPE
        DESCRIPTION.
    sf_to_ref_T : np.ndarray
        DESCRIPTION.
    pix_to_ref_T : np.ndarray
        DESCRIPTION.
    center_xy : list
        DESCRIPTION.
    pixelresolution_xy : list
        DESCRIPTION.

    Returns
    -------
    points_in_scanfield : TYPE
        DESCRIPTION.

    """
    if len(point_to_transform) == 0:
        return np.array([])

    point_to_transform = np.array(point_to_transform)
    if point_to_transform.ndim == 1:
        point_to_transform = point_to_transform.reshape(1, -1)

    if point_to_transform.shape[1] != 2:
        raise ValueError("point_to_transform should have shape (n_points, 2)")

    # From pixel space to reference space
    transformed_pt = np.dot(point_to_transform, pix_to_ref_T.T[:-1, :-1])

    # From reference space to scanner space
    center_pt = [0.5, 0.5]  # Generic center of normalized reference space
    center_pt_ref = np.dot(center_pt, sf_to_ref_T.T[:-1, :])

    # Calculate ROI center translation compared to center
    dx = center_xy[0] - center_pt_ref[0]
    dy = center_xy[1] - center_pt_ref[1]

    # Generate translation matrix
    T_translate = np.array([[1, 0, dx],
                            [0, 1, dy],
                            [0, 0, 1]])

    # Apply translation
    points_in_scanfield = np.dot(
        np.column_stack((transformed_pt,
                         np.ones(transformed_pt.shape[0]))),
        T_translate.T)[:, :2]

    return points_in_scanfield


def tf_transform(
        point_to_transform: tf.Tensor,
        sf_to_ref_T: tf.Tensor,
        pix_to_ref_T: tf.Tensor,
        center_xy: tf.Tensor,
        pixelresolution_xy: tf.Tensor
) -> tf.Tensor:
    """
    Transforms pixel coordinates directly to scanner space coordinates.

    This function accounts for different scanner (ResScan / LinScan)
    field-of-views (FOVs) by using a common normalized reference space.

    The reference space has X and Y coordinates ranging from 0 to 1,
    representing the maximum extents of the scanners
    while maintaining their true aspect ratios.

    The scanner space is mapped into the common reference space
    via affine transformation.
    All Scanfields (including RotatedRectangle) defined within scanner space
    are mapped to reference space via affine transformation.

    Scanfields of type RotatedRectangle have two associated
    affine matrices that allow coordinate space conversions:
        - pixelToRefTransfrom: transform pixel coordinates to reference space
        - affine: transform scanfield coordinates to reference space.

    The inverted matrix T^-1 allows for the opposite transformation.

    Parameters
    ----------
    point_to_transform : tf.Tensor
        Tensor of points to transform.
    sf_to_ref_T : tf.Tensor
        Transformation matrix from scanfield to reference space.
    pix_to_ref_T : tf.Tensor
        Transformation matrix from pixel to reference space.
    center_xy : tf.Tensor
        Center coordinates in the XY plane.
    pixelresolution_xy : tf.Tensor
        Pixel resolution in the XY plane.

    Returns
    -------
    tf.Tensor
        Transformed points in the scanfield.
    """
    if tf.shape(point_to_transform)[0] == 0:
        return tf.constant([], shape=(0, 2), dtype=tf.float32)

    point_to_transform = tf.convert_to_tensor(point_to_transform, dtype=tf.float32)
    if point_to_transform.ndim == 1:
        point_to_transform = tf.reshape(point_to_transform, (1, -1))

    if tf.shape(point_to_transform)[1] != 2:
        raise ValueError("point_to_transform should have shape (n_points, 2)")

    # From pixel space to reference space
    transformed_pt = tf.matmul(point_to_transform, pix_to_ref_T[:-1, :-1], transpose_b=True)

    # From reference space to scanner space
    center_pt = tf.constant([[0.5, 0.5, 1.0]], dtype=tf.float32)  # Include homogeneous coordinate
    center_pt_ref = tf.matmul(center_pt, sf_to_ref_T, transpose_b=True)

    # Calculate ROI center translation compared to center
    dx = center_xy[0] - center_pt_ref[0, 0]
    dy = center_xy[1] - center_pt_ref[0, 1]

    # Generate translation matrix
    T_translate = tf.convert_to_tensor([[1, 0, dx],
                                        [0, 1, dy],
                                        [0, 0, 1]], dtype=tf.float32)

    # Apply translation
    ones = tf.ones((tf.shape(transformed_pt)[0], 1), dtype=tf.float32)
    transformed_pt = tf.concat([transformed_pt, ones], axis=1)
    points_in_scanfield = tf.matmul(transformed_pt, T_translate, transpose_b=True)[:, :2]

    return points_in_scanfield


def mean_and_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    ci = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, ci

