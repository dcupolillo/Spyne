from spyne.core.spines.analysis.segmentation.detection import inference
from spyne.core.spines.analysis.segmentation.post_processing import (
    process_predictions)
from spyne.core.spines.analysis.segmentation.padding import pad_images
from spyne.core.spines.analysis.spatial_distances import (
    euclidean_distance, find_closest_node, distance_along_neurite, find_root,
    prepare_neurite_distance_tools)
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from itertools import chain


def run_inference_and_post_processing(
    images: np.ndarray or list,
    spine_datasets: np.ndarray or list,
    original_dimensions: tuple or list,
    config: dict,
) -> tuple:
    """
    Run the semantic segmentation pipeline: inference and post-processing.

    Parameters
    ----------
    images : np.ndarray or list
        Batch of ROI images for inference.
    spine_datasets : np.ndarray or list
        Batch of RoiSpine instances corresponding to the ROIs.
    original_dimensions : tuple or list
        Original dimensions of the images before padding.
    config : dict
        Configuration dictionary containing model path and parameters for segmentation.

        Expected keys:
        - device : str
            Type of device to use for computation.
        - segmentation_model_fn : str
            Path to the deep learning model for segmentation.
        - spine_threshold : float
            Threshold value to identify spines.
        - dendrite_threshold : float
            Threshold value to identify dendrites.
        - mask_size : int
            Size of the mask used for certain morphological operations.
        - min_distance : int
            Minimum distance between detected centroids.
        - min_dendrite_size : float
            Minimum size to keep dendrites.
        - dendrite_dilation_iterations : int
            Number of iterations for dilating dendrites.
        - min_spine_size : float
            Minimum size to keep spines.

    Returns
    -------
    tuple
        Processed predictions for spines and dendrites, and raw predictions.
    """

    with tqdm(total=2, desc="Inference Progress", leave=True) as pbar:

        # Step 1: Perform inference
        raw_predictions = inference(
            images=images,
            model_fn=config['segmentation_model_fn'],
            original_dimensions=original_dimensions,
            device=config['device'],
        )
        pbar.update(1)

        spine_predictions, dendrite_predictions = raw_predictions

        # Step 2: Process predictions
        processed_predictions = process_predictions(
            spine_datasets=spine_datasets,
            spine_predictions=spine_predictions,
            dendrite_predictions=dendrite_predictions,
            config=config
        )
        pbar.update(1)

    return processed_predictions, raw_predictions


def semantic_segmentation_pipeline(
        dataset,
        spine_dataset,
        config: dict,
) -> tuple:
    """
    Perform semantic segmentation pipeline for a dataset.
    Collects all images and spine dataset managers, then
    pads all images to equal size and finally computes the predictions
    and cleans them with post processing.

    Parameters
    ----------
    dataset : object
        The dataset containing ROIs for segmentation.
    spine_dataset : object
        The SpineDataset object to process each ROI.
    config : dict
        Configuration dictionary containing parameters
        for segmentation and inference.

        Expected keys:
        - device : str
            Type of device to use for computation.
        - segmentation_model_fn : str
            Path to the segmentation model.
        - spine_threshold : float
            Threshold value to identify spines.
        - dendrite_threshold : float
            Threshold value to identify dendrites.
        - mask_size : int
            Size of the mask for morphological operations.
        - min_distance : int
            Minimum distance between detected centroids.
        - min_dendrite_size : float
            Minimum size to keep dendrites.
        - dendrite_dilation_iterations : int
            Number of iterations for dilating dendrites.
        - min_spine_size : float
            Minimum size to keep spines.
        - sd_factor : int
            Multiplier of the standard deviation to use as a binary threshold.

    Returns
    -------
    tuple
        A tuple containing:
        - spine_datasets : list
            List of RoiSpine instances.
        - spines_data : list
            Processed spine data for all ROIs.
        - dendrites_data : list
            Processed dendrite data for all ROIs.
        - spine_predictions : list
            Raw neural network predictions for spine segmentation.
        - dendrite_predictions : list
            Raw neural network predictions for dendrite segmentation.
    """

    images = [None] * len(dataset)
    spine_datasets = [None] * len(dataset)

    morph = dataset._morph

    # Step 1: Collect images and initialize RoiSpine instances
    with tf.device(config['device']):

        for roi_index in tqdm(
                range(len(dataset)),
                desc="Collecting images",
                total=len(dataset)):

            roi_spine = spine_dataset[roi_index]  # RoiSpine instance
            spine_datasets[roi_index] = roi_spine
            images[roi_index] = roi_spine.get_base_image()

    # Step 2: Pad images for consistent dimensions
    padded_images, original_dimensions = pad_images(images)
    padded_images = np.stack(padded_images, axis=0)

    # Step 3: Run the segmentation pipeline
    processed_predictions, raw_predictions = run_inference_and_post_processing(
        padded_images,
        spine_datasets,
        original_dimensions,
        config
    )

    spines_data, dendrites_data = processed_predictions
    spine_predictions, dendrite_predictions = raw_predictions

    # Flatten spines and dendrites data
    spines_data = list(chain.from_iterable(spines_data))
    dendrites_data = list(chain.from_iterable(dendrites_data))

    # Precompute distance calculation tools for efficiency
    node_map, paths_to_root = prepare_neurite_distance_tools(morph.neuron)

    # Add an overall spine_index key and other metadata to the list
    for i, spine_dict in enumerate(spines_data):
        spine_dict['spine_index'] = i

        closest_node_id = find_closest_node(spine_dict, morph.neuron)
        root_node_index = find_root(morph.neuron, closest_node_id)
        
        # find_root returns an index, convert to node ID
        root_node_id = morph.neuron[root_node_index].id

        closest_node_index = closest_node_id - 1
        closest_node = morph.neuron[closest_node_index]

        distance_from_root = distance_along_neurite(
            node_map,
            paths_to_root,
            closest_node_id,
            root_node_id
        )

        spine_position = (
            spine_dict["centroid_fov_um"][0],
            spine_dict["centroid_fov_um"][1],
            spine_dict["roi_z"])

        closest_node_position = (
            closest_node.x, closest_node.y, closest_node.z)

        distance_from_shaft = euclidean_distance(
            spine_position,
            closest_node_position
            )

        spine_dict['distance_from_root'] = distance_from_root
        spine_dict['root_node_id'] = root_node_id
        spine_dict['closest_node_id'] = closest_node_id
        spine_dict['distance_from_shaft'] = distance_from_shaft

    return (
        spine_datasets,
        spines_data,
        dendrites_data,
        spine_predictions,
        dendrite_predictions)
