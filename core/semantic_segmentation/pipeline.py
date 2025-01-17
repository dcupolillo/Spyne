from spyne.core.semantic_segmentation.inference import inference
from spyne.core.semantic_segmentation.post_processing import process_predictions
from spyne.core.semantic_segmentation.padding import pad_images
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from itertools import chain
from pathlib import Path
import flammkuchen as fl


def run_inference_and_post_processing(
    images: np.ndarray or list,
    segmenters: np.ndarray or list,
    original_dimensions: tuple or list,
    config: dict,
) -> tuple:
    """
    Run the semantic segmentation pipeline: inference and post-processing.

    Parameters
    ----------
    images : np.ndarray or list
        Batch of ROI images for inference.
    segmenters : np.ndarray or list
        Batch of segmenters or masks corresponding to the ROIs.
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
        - kernel_size : int
            Size of the structuring element used for morphological operations.

    Returns
    -------
    tuple
        Processed predictions for spines and dendrites.
    """

    with tqdm(total=2, desc="Inference Progress", leave=True) as pbar:

        # Step 1: Perform inference
        spine_predictions, dendrite_predictions = inference(
            images=images,
            model_fn=config['segmentation_model_fn'],
            original_dimensions=original_dimensions,
            device=config['device'],
        )
        pbar.update(1)

        # Step 2: Process predictions
        processed_predictions = process_predictions(
            segmenters=segmenters,
            spine_predictions=spine_predictions,
            dendrite_predictions=dendrite_predictions,
            config=config
        )
        pbar.update(1)

    return processed_predictions


def semantic_segmentation_pipeline(
        dataset,
        segmenter,
        config: dict,
        output_folder: str or Path
) -> tuple:
    """
    Perform semantic segmentation pipeline for a dataset.
    Collects all simages and segmenter managers, then
    pad all images to equal size and finally compute the predictions
    and cleans them with post processing.

    Parameters
    ----------
    dataset : object
        The dataset containing ROIs for segmentation.
    segmenter : object
        The RoiSegmenter object to process each ROI.
    device : str
        The device to run the computation on (e.g., '/GPU:0' or '/CPU:0').
    config : dict
        Configuration dictionary containing parameters for segmentation and inference.

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
        - kernel_size : int
            Size of the structuring element for morphological operations.
        - sd_factor : int
            Multiplier of the standard deviation to use as a binary threshold.

    Returns
    -------
    tuple
        A tuple containing:
        - spines_data : list
            Processed spine data for all ROIs.
        - dendrites_data : list
            Processed dendrite data for all ROIs.
    """

    output_folder = Path(output_folder)
    images = [None] * len(dataset)
    segmenters = [None] * len(dataset)

    # Step 1: Collect images and initialize RoiSegmenter instances
    with tf.device(config['device']):

        for roi_index in tqdm(
                range(len(dataset)),
                desc="Collecting images",
                total=len(dataset)):

            roi_segmenter = segmenter[roi_index]  # RoiSegmenter instanc
            segmenters[roi_index] = roi_segmenter
            images[roi_index] = roi_segmenter.get_base_image()

    # Step 2: Pad images for consistent dimensions
    padded_images, original_dimensions = pad_images(images)
    padded_images = np.stack(padded_images, axis=0)

    # Step 3: Run the segmentation pipeline
    spines_data, dendrites_data = run_inference_and_post_processing(
        padded_images,
        segmenters,
        original_dimensions,
        config
    )

    # Flatten spines and dendrites data
    spines_data = list(chain.from_iterable(spines_data))
    dendrites_data = list(chain.from_iterable(dendrites_data))

    # Optionally save precomputed data
    fl.save(Path(output_folder, "spines_data.h5"), spines_data)

    return segmenters, spines_data, dendrites_data
