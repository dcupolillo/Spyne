""" Created on Mon Oct 23 12:23:47 2023
    @author: dcupolillo """

from pathlib import Path
from typing import Union, List, Tuple, Dict, Any

import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, moments
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import morphology
from scipy import ndimage
from tensorflow.keras.models import load_model
from spyne.core.utils.utils import transform


def pad_image(
        image: np.ndarray,
        target_height: int,
        target_width: int
) -> np.ndarray:
    """
    Helper function to pad a single
    image to the target height and width.
    """

    height, width = image.shape
    pad_height = target_height - height
    pad_width = target_width - width
    padded_image = np.pad(
        image,
        ((0, pad_height), (0, pad_width)),
        mode='reflect')

    return padded_image


def pad_images(
        images: np.ndarray or list(np.ndarray),
        pad_size: int = 32
) -> np.ndarray or list(np.ndarray):
    """
    Pad a list of images or a single image to meet size requirements
    for the neural network model.

    Parameters
    ----------
    images : np.ndarray or list(np.ndarray)
        The list of input images or a single image.
    pad_size : int, optional
        The size to pad to (default is 32).

    Returns
    -------
    np.ndarray or list(np.ndarray)
        The list of padded images or a single padded image.
    """

    if isinstance(images, list):
        # Find the maximum height and width
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)

        # Ensure the dimensions are multiples of pad_size
        target_height = (max_height + pad_size - 1) // pad_size * pad_size
        target_width = (max_width + pad_size - 1) // pad_size * pad_size

        padded_images = [
            pad_image(image, target_height, target_width)
            for image in images]
        original_dimensions = [
            (image.shape[0], image.shape[1]) for image in images]

    else:
        # Single image case
        height, width = images.shape
        target_height = (height + pad_size - 1) // pad_size * pad_size
        target_width = (width + pad_size - 1) // pad_size * pad_size
        padded_images = pad_image(images, target_height, target_width)
        original_dimensions = (height, width)

    return padded_images, original_dimensions


def unpad_prediction(
        pred: np.ndarray,
        original_dimensions: Tuple,
) -> np.ndarray:
    """
    Helper function to unpad a single
    image to restore the original height and width.
    """

    original_height, original_width = original_dimensions

    return pred[:original_height, :original_width]


def unpad_images(
        predictions: Union[List[np.ndarray], np.ndarray],
        original_dimensions: Union[Tuple[int, int], List[Tuple[int, int]]]
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Unpad a list of predictions or a single prediction based on the pad factor.

    Parameters
    ----------
    predictions : Union[List[np.ndarray], np.ndarray]
        The list of predictions or a single prediction.
    pad_size : int, optional
        The size to unpad (default is 32).

    Returns
    -------
    Union[List[np.ndarray], np.ndarray]
        The list of unpadded predictions or a single unpadded prediction.
    """
    if predictions.ndim != 2:
        unpadded_images = [
            unpad_prediction(pred, dims)
            for pred, dims in zip(predictions, original_dimensions)]

    else:
        unpadded_images = unpad_prediction(
            predictions, original_dimensions)

    return unpadded_images


def inference(
        images: np.ndarray,
        model_fn: Union[str, Path],
        original_dimensions: Union[Tuple[int, int], List[Tuple[int, int]]],
        pad_size=32,
) -> (np.ndarray, np.ndarray):
    """
    Load the model and run inference to get raw predictions.
    Handles both single images and batches: if images is a list,
    it is stacked into a 3D array (batch), whereas if images is a 2D array,
    it is expanded to a 3D array by adding a batch dimension.

    Parameters
    ----------
    images : Union[np.ndarray, List[np.ndarray]]
        The input images for segmentation. Can be a single image (2D array)
        or a batch of images (3D array or list of 2D arrays).
        Must be already padded.
    model_fn : str
        Path to the pre-trained model.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Raw spine and dendrite predictions.
    """

    if images.ndim == 2:
        images = images[np.newaxis, ...]

    # Normalize the input image to have values between -1 and 1.
    images_min = images.min(axis=(1, 2), keepdims=True)
    images_max = images.max(axis=(1, 2), keepdims=True)
    images = (images - images_min) / (images_max - images_min) * 2 - 1

    # Load the model and run inference
    model_path = Path(__file__).resolve().parent.parent / model_fn
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Run inference on GPU
    with tf.device('/gpu:0'):
        model = load_model(model_path, compile=False)
        pd, ps = model.predict(images[..., None])

    ps = ps.squeeze()
    pd = pd.squeeze()

    spine_predictions = unpad_images(ps, original_dimensions)
    dendrite_predictions = unpad_images(pd, original_dimensions)

    if ps.ndim == 2:
        # Unpad images to original dimensions
        spine_predictions = unpad_images(ps, original_dimensions)
        dendrite_predictions = unpad_images(pd, original_dimensions)

    return spine_predictions, dendrite_predictions


def process_single_prediction(
        roi: object,
        spine_prediction: np.ndarray,
        dendrite_prediction: np.ndarray,
        spine_threshold: float,
        dendrite_threshold: float,
        min_spine_size: float,
        mask_size: int,
        min_distance: int,
        min_dendrite_size: float,
        dendrite_dilation_iterations: int,
        kernel_size: int,
        sd_factor: int,
) -> Tuple[
        Dict[Tuple[int, int], Dict[int, Dict[str, Any]]],
        Dict[Tuple[int, int], Dict[int, Dict[str, Any]]]
]:
    """
    Perform semantic segmentation to identify spines and dendrites.

    Parameters
    ----------
    roi : object (spyne.ImagingDataset.RoiGroup.Roi)
        Region of interest.
    base_image : np.ndarray
        The input image for segmentation.
    model_fn : str
        Path to the pre-trained model.
    spine_threshold : float
        Threshold value to identify spines.
    dendrite_threshold : float
        Threshold value to identify dendrites.
    min_spine_size : float
        Minimum size to keep spines.
    mask_size : int
        The size of the mask used for some operations.
    min_distance : int
        Minimum distance between detected centroids.
    min_dendrite_size : float
        Minimum size to keep dendrites.
    dendrite_dilation_iterations: int
        Number of iterations for dilating dendrites.
    kernel_size : int
        Size of the structuring element used for morphological operations.
    sd_factor: int
        Multiplier of SD to use as binary threshold.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        List of dictionaries containing spine information and
        dendrite information.
    """

    # Apply thresholds to generate binary masks
    spine_mask = threshold_prediction(
        spine_prediction, spine_threshold)
    dendrite_mask = threshold_prediction(
        dendrite_prediction, dendrite_threshold)

    # Labels same-value neighbors pixels
    spine_labels = label(spine_mask)
    dendrite_labels = label(dendrite_mask)

    # Remove corner joints from the spine mask
    spine_labels = remove_corner_joints(spine_labels)

    # Removes spine labels too distant from dendrite
    spine_labels = remove_distant_spines(
        spine_labels,
        dendrite_mask,
        dendrite_dilation_iterations)

    # Remove small labeled spines and dendrites
    # HINT : it might be an error to remove small spines?
    spine_labels = remove_small_labels(
        spine_labels,
        min_spine_size)
    dendrite_labels = remove_small_labels(
        dendrite_labels,
        min_dendrite_size)

    # Re-create binary mask from cleaned spine segmentation
    spine_mask = np.where(spine_labels > 0, 255, 0).astype(np.uint8)
    dendrite_mask = np.where(dendrite_labels > 0, 255, 0).astype(np.uint8)

    # TODO: fill holes?

    # Create a distance map for each pixel to the nearest background pixel
    # cv2.DIST_L2 specifies Euclidean distance
    distance_map = cv2.distanceTransform(
        spine_mask,
        cv2.DIST_L2,
        mask_size)

    local_maxima = peak_local_max(
        distance_map,
        footprint=np.ones((3, 3)),
        min_distance=min_distance,
        labels=spine_mask,
        exclude_border=False)

    markers = np.zeros_like(spine_mask, dtype=np.uint8)

    # Place markers at the coordinates of the local maxima
    for x, y in local_maxima:
        markers[x, y] = 1

    # Apply watershed
    # TODO: check these parameters
    watershed_spines = watershed(
        -distance_map,
        markers,
        mask=spine_mask,
        connectivity=np.ones((3, 3), dtype=bool),  # cross (+) neighbors
        compactness=0.01)

    # TODO: implement dilation of single-pixel markers?
    # spine_labels_test = morphology.isotropic_dilation(
    #     markers, radius=1)

    # Re-label components
    spine_labels = label(watershed_spines)
    dendrite_labels = label(dendrite_mask)

    # remove 0 label because it is background
    unique_spine_labels = np.unique(spine_labels)[1:]
    unique_dendrite_labels = np.unique(dendrite_labels)[1:]

    spine_single_labels = np.array([None] * len(unique_spine_labels))
    dendrite_single_labels = np.array([None] * len(unique_dendrite_labels))

    # Re-derive centroids from the segmented regions post-watershed
    centroids = np.array([None] * len(unique_spine_labels))

    for n in unique_spine_labels:
        cy, cx = calculate_centroid(spine_labels == n)
        centroids[n-1] = [cx, cy]
        binary_mask = (spine_labels == n).astype(np.uint8)
        if np.any(binary_mask):
            spine_single_labels[n-1] = binary_mask

    # Initialize the nested dictionaries
    spines_dicts = [None] * len(spine_single_labels)
    dendrites_dicts = [None] * len(unique_dendrite_labels)

    # Populate the spines dictionary
    for spine_n, spine in enumerate(spine_single_labels):
        if spine is not None:
            spines_dicts[spine_n] = {
                'roi_n': roi.roi_index,
                'roi_z': roi.z,
                'centroid_pix': np.asarray(centroids[spine_n]),
                'mask': spine
            }

    # Extract individual dendrites
    for n in unique_dendrite_labels:
        binary_mask = (dendrite_labels == n).astype(np.uint8)
        if np.any(binary_mask):
            dendrite_single_labels[n-1] = binary_mask

    # Populate the dendrites dictionary
    for dendrite_n, dendrite in enumerate(dendrite_single_labels):
        if dendrite is not None:
            skeletonized = morphology.skeletonize(dendrite)
            rows, cols = np.nonzero(skeletonized)
            skel_coords = [[c, r] for r, c in zip(rows, cols)]
            dendrites_dicts[dendrite_n] = {
                'roi_n': roi.roi_index,
                'mask': dendrite,
                'skeleton': skel_coords
            }

    # Calculate skeleton coordinates for FOV
    skel_coords_fov = []
    for dendrite in dendrites_dicts:
        skeleton = dendrite['skeleton']
        coords = [[c, r] for r, c in skeleton]
        skel_coords_fov.append(transform(
            coords,
            np.array(roi.affine),
            np.array(roi.pixel_to_ref_transform),
            roi.center_xy,
            roi.pixel_resolution_xy))

    # Update dendrites dictionary with skeleton FOV coordinates
    for dendrite_n, node in enumerate(skel_coords_fov):
        dendrites_dicts[dendrite_n]['skeleton_fov'] = node.tolist()
    # dendrites_dicts[key] = [{**dendrite, 'skeleton_fov': node.tolist()}
    #                         for dendrite, node in zip(
    #                                 dendrites_dicts[key].values(),
    #                                 skel_coords_fov)]

    centroids_fov = transform(
        [spine['centroid_pix']
         for spine in spines_dicts],
        np.array(roi.affine),
        np.array(roi.pixel_to_ref_transform),
        roi.center_xy,
        roi.pixel_resolution_xy)

    spines_dicts = [{**spine, 'centroid_fov': centroid.tolist()}
                    for spine, centroid in zip(
                        spines_dicts, centroids_fov)]

    # spines_dicts = [{**spine, 'centroid_fov_um': centroid.tolist()}
    #                 for spine, centroid in zip(
    #                     spines_dicts, centroids_fov)]


    return spines_dicts, dendrites_dicts


def process_predictions(
        segmenters: np.ndarray,
        spine_predictions: np.ndarray,
        dendrite_predictions: np.ndarray,
        spine_threshold: float,
        dendrite_threshold: float,
        min_spine_size: float,
        mask_size: int,
        min_distance: int,
        min_dendrite_size: float,
        dendrite_dilation_iterations: int,
        kernel_size: int,
        sd_factor: int,
) -> Tuple[
        Dict[Tuple[int, int], Dict[int, Dict[str, Any]]],
        Dict[Tuple[int, int], Dict[int, Dict[str, Any]]]
]:
    """
    Perform semantic segmentation to identify spines and dendrites
    for single or batch predictions.

    This function processes the predictions from the neural network to identify
    spines and dendrites within a single ROI or a batch of ROIs.
    It applies various image processing techniques to refine the predictions
    and generate final segmentation results.

    Parameters
    ----------
    segmenters : np.ndarray
        A single RoiSegmenter instance or a list of RoiSegmenter instances
        representing the regions of interest (ROIs).
    spine_predictions : np.ndarray
        The raw spine predictions from the neural network.
        Can be a single prediction or a batch of predictions.
    dendrite_predictions : np.ndarray
        The raw dendrite predictions from the neural network.
        Can be a single prediction or a batch of predictions.
    spine_threshold : float
        The threshold value to identify spines.
    dendrite_threshold : float
        The threshold value to identify dendrites.
    min_spine_size : float
        The minimum size to keep spines.
    mask_size : int
        The size of the mask used for certain operations.
    min_distance : int
        The minimum distance between detected
        centroids.
        min_dendrite_size : float
        The minimum size to keep dendrites.
        dendrite_dilation_iterations : int
        The number of iterations for dilating dendrites.
        kernel_size : int
        The size of the structuring element used for morphological operations.
        sd_factor : int
        Multiplier of the standard deviation to use as a binary threshold.

    Returns
    -------
    Union[Tuple[List[dict], List[dict]],
          Tuple[List[List[dict]], List[List[dict]]]]
        If processing a single ROI, returns a tuple of lists containing
        dictionaries with spine and dendrite information.
        If processing a batch of ROIs, returns a tuple of lists of lists
        containing dictionaries with spine and dendrite information
        for each ROI.

    Notes
    -----
    The function first checks if the predictions are for a batch of ROIs or a
    single ROI. It then processes each prediction accordingly
    by calling `process_single_prediction` for each pair of
    spine and dendrite predictions. The resulting segmentation information
    is compiled into dictionaries which include details about
    the identified spines and dendrites.
    """

    if isinstance(spine_predictions, list):

        batch_spines_dicts = []
        batch_dendrites_dicts = []

        for spine_pred, dendrite_pred, roi in zip(
                spine_predictions, dendrite_predictions, segmenters):

            spines_dicts, dendrites_dicts = process_single_prediction(
                roi=roi,
                spine_prediction=spine_pred,
                dendrite_prediction=dendrite_pred,
                spine_threshold=spine_threshold,
                dendrite_threshold=dendrite_threshold,
                min_spine_size=min_spine_size,
                mask_size=mask_size,
                min_distance=min_distance,
                min_dendrite_size=min_dendrite_size,
                dendrite_dilation_iterations=dendrite_dilation_iterations,
                kernel_size=kernel_size,
                sd_factor=sd_factor,)
            batch_spines_dicts.append(spines_dicts)
            batch_dendrites_dicts.append(dendrites_dicts)

        return batch_spines_dicts, batch_dendrites_dicts

    else:
        print("Processing single prediction..")
        return process_single_prediction(
            roi=segmenters,
            spine_prediction=spine_predictions,
            dendrite_prediction=dendrite_predictions,
            spine_threshold=spine_threshold,
            dendrite_threshold=dendrite_threshold,
            min_spine_size=min_spine_size,
            mask_size=mask_size,
            min_distance=min_distance,
            min_dendrite_size=min_dendrite_size,
            dendrite_dilation_iterations=dendrite_dilation_iterations,
            kernel_size=kernel_size,
            sd_factor=sd_factor)


def threshold_prediction(
        prediction: np.ndarray,
        threshold: float
) -> np.ndarray:

    return (prediction > threshold).astype(np.uint8) * 255


def remove_distant_spines(
        labels: np.ndarray,
        reference_prediction: np.ndarray,
        iterations: int
) -> np.ndarray:

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
    Remove objects smaller than the specified minimum size from labeled images.

    Parameters:
    ----------
    labels : np.ndarray
        The labeled image array where each unique value
        represents a different object.
    min_size : int
        The minimum size threshold for objects to be retained.

    Returns:
    -------
    labels : np.ndarray
        The modified labeled image with small objects removed.
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
    """ Define the 2x2 patterns to identify (non-zero values)."""
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
    Remove corner joints in a labeled mask by identifying and modifying
    specific 2x2 patterns where non-zero values represent labels.

    Parameters
    ----------
    mask : np.ndarray
        Labeled mask with connected components.

    Returns
    -------
    np.ndarray
        Modified labeled mask with corner joints removed.
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


def calculate_intersection(
        mask1: np.ndarray,
        mask2: np.ndarray
) -> int:
    """
    Calculate the number of overlapping pixels between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask.
    mask2 : np.ndarray
        Second binary mask.

    Returns
    -------
    int
        Number of overlapping pixels between the two masks.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("The two masks must have the same shape")

    # Calculate intersection
    intersection = np.logical_and(mask1, mask2)

    # Return the count of overlapping pixels
    return np.sum(intersection)


def create_intersection_matrix(
        sd_masks: list[np.ndarray],
        prediction_masks: list[np.ndarray]
) -> tuple:
    """
    Create an intersection matrix between two groups of binary masks and return
    the indices of pairs of masks with an intersection greater than 0.

    Parameters
    ----------
    sd_masks : list of np.ndarray
        List of binary masks from the first group.
    prediction_masks : list of np.ndarray
        List of binary masks from the second group.

    Returns
    -------
    tuple
        A tuple containing:
        - np.ndarray: A matrix where element (i, j) is
            the number of overlapping
            pixels between sd_masks[i] and prediction_masks[j].
        - list of tuple: List of tuples where each tuple contains the indices
          (i, j) of the pairs of masks that have an overlap greater than 0.
    """
    num_sd_masks = len(sd_masks)
    num_prediction_masks = len(prediction_masks)

    intersection_matrix = np.zeros(
        (num_sd_masks, num_prediction_masks), dtype=int)
    overlap_indices = []

    for i, sd_mask in enumerate(sd_masks):
        for j, prediction_mask in enumerate(prediction_masks):
            intersection = calculate_intersection(sd_mask, prediction_mask)
            intersection_matrix[i, j] = intersection
            if intersection > 0:
                overlap_indices.append((i, j))

    return intersection_matrix, overlap_indices


def create_sd_masks(
        roi: object,
        kernel_size: int,
        sd_factor: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Create standard deviation masks for the region of interest (ROI)
    and compute centroids and binary masks for each connected component.

    Parameters
    ----------
    roi : list
        List containing ROI data.
    kernel_size : int
        Size of the structuring element used for morphological operations.
    sd_factor: float
        Multiplier of SD to use as binary threshold

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        A tuple containing:
        - sd_centroids: An array of centroids.
        - spine_single_labels: A list of binary masks.
    """

    # half_sd_window = int(int(roi.frame_rate * 1) / 2)
    # stim_frame = int(roi.frame_rate * 1)
    # indexes_around_stim = np.arange(stim_frame - half_sd_window,
    #                                 stim_frame + half_sd_window)

    # Create binary masks based on standard deviation threshold
    # sd_masks = []
    # for n in roi.sweep_list:
    #     frames_around_stim = np.array(
    #         [roi[n].ch1[i].frame for i in indexes_around_stim])
    #     std_proj_around_stim = np.std(frames_around_stim, axis=0)
    #     sd_mask = (std_proj_around_stim >
    #                np.mean(std_proj_around_stim) * sd_factor)
    #     sd_masks.append(sd_mask)

    sd_masks = [(roi[n].ch1.std.frame >
                 np.mean(roi[n].ch1.std.frame) * sd_factor)
                for n in roi.sweep_list]

    # Combine masks to get the maximum mask
    sd_mask_max = np.max(sd_masks, axis=0)

    # Morphological operations
    selem = morphology.square(kernel_size)
    sd_mask_max = morphology.binary_dilation(sd_mask_max, selem)
    sd_mask_max = morphology.binary_erosion(sd_mask_max, selem)

    # Remove small objects
    labeled_array, num_features = ndimage.label(sd_mask_max)
    component_sizes = ndimage.sum(
        sd_mask_max, labeled_array, range(num_features + 1))
    size_mask = component_sizes > 1
    size_mask[0] = 0  # Keep background labeled as 0

    cleaned_eroded = size_mask[labeled_array]
    sd_mask_max = cleaned_eroded.astype(bool)

    # Label connected components
    sd_mask_max_labeled = label(sd_mask_max)
    unique_spine_labels = np.unique(sd_mask_max_labeled)[1:]

    # Initialize arrays for centroids and binary masks
    centroids = []
    spine_single_labels = []

    for n in unique_spine_labels:
        binary_mask = (sd_mask_max_labeled == n).astype(np.uint8)
        if np.any(binary_mask):
            cy, cx = calculate_centroid(binary_mask)
            centroids.append([cx, cy])
            spine_single_labels.append(binary_mask)

    return np.array(centroids), spine_single_labels


def move_mask_to_sd_centroid(
        mask: np.ndarray,
        current_centroid: Tuple[int, int],
        new_centroid: Tuple[int, int]
) -> np.ndarray:
    """
    Move a binary mask to a new centroid position.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with a single component.
    current_centroid : Tuple[int, int]
        Current centroid (y, x) of the mask.
    new_centroid : Tuple[int, int]
        New centroid (y, x) to which the mask should be moved.

    Returns
    -------
    np.ndarray
        Binary mask moved to the new centroid.
    """
    # Calculate the shift required
    shift_y = new_centroid[0] - current_centroid[0]
    shift_x = new_centroid[1] - current_centroid[1]

    # Apply the shift to the mask
    shifted_mask = ndimage.shift(
        mask.astype(float),
        shift=(shift_y, shift_x),
        order=0,
        mode='constant',
        cval=0)

    return shifted_mask > 0


def calculate_centroid(
        mask: np.ndarray
) -> Tuple[int, int]:

    M = moments(mask)
    cy, cx = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]

    return (cy, cx)
