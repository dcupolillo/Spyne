""" Created on Mon Oct 23 12:23:47 2023
    @author: dcupolillo """

import numpy as np
from spyne.core.semantic_segmentation.utils import (
    threshold_prediction, remove_corner_joints,
    remove_distant_spines, remove_small_labels,
    calculate_centroid)
from spyne.core.utils.utils import transform
from skimage import morphology
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import cv2


def process_prediction(
        roi: object,
        spine_prediction: np.ndarray,
        dendrite_prediction: np.ndarray,
        config: dict,
) -> tuple:
    """
    This function processes raw neural network predictions to segment spines and
    dendrites in a single region of interest (ROI). It applies thresholds,
    morphological operations, and centroid calculations to refine the segmentation.

    Building block for process_predictions().

    Parameters
    ----------
    roi : object
        Single region of interest object containing metadata
    spine_prediction : np.ndarray
        Neural network prediction array for spine segmentation.
    dendrite_prediction : np.ndarray
        Neural network prediction array for dendrite segmentation.
    config : dict
        Dictionary containing all configuration parameters required for processing:
        - spine_threshold : float
            Threshold value for binarizing spine predictions.
        - dendrite_threshold : float
            Threshold value for binarizing dendrite predictions.
        - min_spine_size : float
            Minimum size for retaining spines after segmentation.
        - mask_size : int
            Size of the mask used for distance transforms.
        - min_distance : int
            Minimum distance between detected spine centroids.
        - min_dendrite_size : float
            Minimum size for retaining dendrites after segmentation.
        - dendrite_dilation_iterations : int
            Number of dilation iterations to expand dendrite regions.
        - kernel_size : int
            Size of the structuring element used for morphological operations.

    Returns
    -------
    tuple
        - spines_dicts : list
            List of dictionaries containing spine information.
        - dendrites_dicts : list
            List of dictionaries containing dendrite information.

    Notes
    -----
    - The function includes steps for binarizing predictions, removing small
      objects, computing centroids, and generating skeletonized representations
      for dendrites.
    - The watershed algorithm is used to separate closely connected spines.
    """

    # Retrieve information from config dictionary
    spine_threshold = config["spine_threshold"]
    dendrite_threshold = config["dendrite_threshold"]
    min_spine_size = config["min_spine_size"]
    mask_size = config["mask_size"]
    min_distance = config["min_distance"]
    min_dendrite_size = config["min_dendrite_size"]
    dendrite_dilation_iterations = config["dendrite_dilation_iterations"]

    # Apply thresholds to generate binary masks
    spine_mask = threshold_prediction(spine_prediction, spine_threshold)
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
    spine_labels = remove_small_labels(spine_labels, min_spine_size)
    dendrite_labels = remove_small_labels(dendrite_labels, min_dendrite_size)

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
                'roi_n': roi.roi_metadata['n_roi'],
                'roi_z': roi.roi_metadata['z'],
                'centroid_pix': np.asarray(centroids[spine_n]),
                'mask': spine,
                'branch_id': roi.roi_metadata['branch_id'],
                'branch_degree': roi.roi_metadata['branch_degree'],
                'spine_area_pix': np.sum(spine),
                'spine_area_um': (
                    np.sum(spine) *
                    roi.roi_metadata['resolution'][0] *
                    roi.roi_metadata['resolution'][1]
                )
            }

    # Extract individual dendrites
    for n in unique_dendrite_labels:
        binary_mask = (dendrite_labels == n).astype(np.uint8)
        if np.any(binary_mask):
            dendrite_single_labels[n-1] = binary_mask

    # FIXME: improve dendritic definition
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

    spines_dicts = [
        {
            **spine,
            'centroid_fov': centroid.tolist(),
            'centroid_fov_um': [
                coord * roi.roi.roi_metadata['objective resolution']
                for coord in centroid.tolist()]
        }
        for spine, centroid in zip(spines_dicts, centroids_fov)
    ]

    # spines_dicts = [{**spine, 'centroid_fov_um': centroid.tolist()}
    #                 for spine, centroid in zip(
    #                     spines_dicts, centroids_fov)]

    return spines_dicts, dendrites_dicts


def process_predictions(
        segmenters: np.ndarray,
        spine_predictions: np.ndarray,
        dendrite_predictions: np.ndarray,
        config: dict,
) -> tuple:
    """
    This function processes multiple predictions obtained from the neural network
    to identify spines and dendrites for single or multiple regions of interest (ROIs).
    For each ROI, it applies segmentation refinement steps using
    process_prediction().

    Parameters
    ----------
    segmenters : np.ndarray
        Array or list of ROI segmenters, each representing an
        individual ROI.
    spine_predictions : np.ndarray
        Neural network predictions for spines.
        Can be a single prediction or a batch of predictions.
    dendrite_predictions : np.ndarray
        Neural network predictions for dendrites.
        Can be a single prediction or a batch of predictions.
    config : dict
        Dictionary containing all configuration parameters required
        for processing:
        - spine_threshold : float
            Threshold value for binarizing spine predictions.
        - dendrite_threshold : float
            Threshold value for binarizing dendrite predictions.
        - min_spine_size : float
            Minimum size for retaining spines after segmentation.
        - mask_size : int
            Size of the mask used for distance transforms.
        - min_distance : int
            Minimum distance between detected spine centroids.
        - min_dendrite_size : float
            Minimum size for retaining dendrites after segmentation.
        - dendrite_dilation_iterations : int
            Number of dilation iterations to expand dendrite regions.
        - kernel_size : int
            Size of the structuring element used for morphological operations.

    Returns
    -------
    tuple
        - List of dictionaries for spines and dendrites for single
        or batch ROIs.

    Notes
    -----
    - When `spine_predictions` and `dendrite_predictions` are lists,
    the function processes them as a batch.
    - For single predictions, it directly calls `process_single_prediction`.
    """

    if isinstance(spine_predictions, list):

        batch_spines_dicts = []
        batch_dendrites_dicts = []

        for spine_pred, dendrite_pred, roi in zip(
                spine_predictions, dendrite_predictions, segmenters):

            spines_dicts, dendrites_dicts = process_prediction(
                roi=roi,
                spine_prediction=spine_pred,
                dendrite_prediction=dendrite_pred,
                config=config)
            batch_spines_dicts.append(spines_dicts)
            batch_dendrites_dicts.append(dendrites_dicts)

        return batch_spines_dicts, batch_dendrites_dicts

    else:
        print("Processing single prediction..")
        return process_prediction(
            roi=segmenters,
            spine_prediction=spine_predictions,
            dendrite_prediction=dendrite_predictions,
            config=config)
