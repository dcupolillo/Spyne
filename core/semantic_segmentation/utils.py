""" Created on Mon Oct 23 12:23:47 2023
    @author: dcupolillo """

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import moments
import tensorflow as tf


def threshold_prediction(
        prediction: np.ndarray,
        threshold: float
) -> np.ndarray:
    """
    Apply a binary threshold to a prediction array.

    Parameters
    ----------
    prediction : np.ndarray
        The input prediction array.
    threshold : float
        The threshold value. Values greater than this will be set to 255, 
        and others to 0.

    Returns
    -------
    np.ndarray
        A binary array where values above the threshold are set to 255 
        and others to 0.

    Notes
    -----
    - This function is commonly used to binarize semantic segmentation outputs.
    """

    return (prediction > threshold).astype(np.uint8) * 255


def remove_distant_spines(
        labels: np.ndarray,
        reference_prediction: np.ndarray,
        iterations: int
) -> np.ndarray:
    """
    Remove labeled pixels in a mask that are not close to a reference mask (dendrite).

    Parameters
    ----------
    labels : np.ndarray
        The labeled mask where each unique integer represents a distinct region.
    reference_prediction : np.ndarray
        The binary reference mask used to determine proximity.
    iterations : int
        The number of dilation iterations applied to the reference mask.

    Returns
    -------
    np.ndarray
        A new labeled mask with distant labels removed.

    Notes
    -----
    - Uses binary dilation on the reference mask to define proximity.
    - Retains only labels that overlap with the dilated reference mask.
    """

    dilation = binary_dilation(
        reference_prediction,
        iterations=iterations)

    unique_labels = np.unique(labels)
    new_labels = np.zeros_like(labels)

    for lbl in unique_labels:
        if lbl == 0:
            continue

        label_mask = (labels == lbl)

        if np.any(dilation & label_mask):
            new_labels[label_mask] = lbl

    return new_labels


def remove_small_labels(
        labels: np.ndarray,
        min_size: int
) -> np.ndarray:
    """
    Remove objects smaller than the specified size from a labeled image.

    Parameters
    ----------
    labels : np.ndarray
        The labeled mask where each unique integer represents a distinct object.
    min_size : int
        The minimum size (in pixels) for an object to be retained.

    Returns
    -------
    np.ndarray
        A modified labeled mask with small objects removed.

    Notes
    -----
    - Objects smaller than `min_size` are set to 0 (background).
    - Label 0 (background) is always preserved.
    """

    # Calculate the size of each unique label
    component_sizes = np.bincount(
        labels.ravel(), minlength=np.max(labels) + 1)

    # Create a mask for small objects, excluding the background (label 0)
    too_small = component_sizes < min_size
    too_small[0] = False

    # Use the mask to set small objects to 0
    labels[too_small[labels]] = 0

    return labels


def is_corner_joint(window):
    """
    Identify 2x2 corner joint patterns in a binary window.

    Parameters
    ----------
    window : np.ndarray
        A 2x2 binary array.

    Returns
    -------
    bool
        True if the window matches a corner joint pattern, otherwise False.

    Notes
    -----
    - Corner joint patterns are configurations where non-zero values form 
      a diagonal without forming a square.
    """
    return ((window[0, 1] != 0 and
             window[1, 0] != 0 and
             window[0, 0] == 0 and
             window[1, 1] == 0) or
            (window[0, 0] != 0 and
             window[1, 1] != 0 and
             window[0, 1] == 0 and
             window[1, 0] == 0))


def remove_corner_joints(mask: np.ndarray) -> np.ndarray:
    """
    Remove corner joints from a labeled mask.

    Parameters
    ----------
    mask : np.ndarray
        A labeled mask with connected components.

    Returns
    -------
    np.ndarray
        The modified mask with corner joints removed.

    Notes
    -----
    - A corner joint is a 2x2 pattern where two diagonal pixels are non-zero, 
      and the other two are zero.
    - Iterates over the mask with a 2x2 moving window and removes detected 
      corner joints.
    """

    # Copy the mask to avoid modifying the original
    modified_mask = mask.copy()

    # Create a list to store the positions where patterns are found
    pattern_positions = []

    # Iterate over the mask using a 2x2 moving window
    for y in range(modified_mask.shape[0] - 1):
        for x in range(modified_mask.shape[1] - 1):
            window = modified_mask[y:y+2, x:x+2]

            if is_corner_joint(window):
                pattern_positions.append((y, x))
                modified_mask[y:y+2, x:x+2] = 0

    return modified_mask


def calculate_centroid(
        mask: np.ndarray
) -> tuple:
    """
    Calculate the centroid of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask representing a single labeled region.

    Returns
    -------
    Tuple[float, float]
        The (y, x) coordinates of the centroid.

    Notes
    -----
    - The centroid is computed using image moments.
    - Assumes that the input mask has non-zero values only in the region 
      of interest.
    """

    M = moments(mask)
    cy, cx = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]

    return (cy, cx)


def transform(
        point_to_transform: list or np.ndarray,
        sf_to_ref_T: np.ndarray,
        pix_to_ref_T: np.ndarray,
        center_xy: list,
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
