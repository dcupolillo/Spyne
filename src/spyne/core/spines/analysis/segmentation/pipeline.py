"""
Module to perform semantic segmentation with either DeepD3 or nnU-Net,
and subsequent post-processing of predictions.

Each segmentation is perform through subprocesses to ensure modularity and 
avoid dependency conflicts between TensorFlow (DeepD3) and PyTorch (nnU-Net).

The main function is `semantic_segmentation_pipeline()`, which orchestrates:
1) Collects images from ROIs (or loads cached denoised images if available)
2) Initializes RoiSpine instances
3) Pads images to consistent dimensions for batch processing
4) Runs inference using the specified algorithm (DeepD3 or nnU-Net)
    - DeepD3 returns probability maps that require thresholding
        - Probability maps are saved as TIFF files in organized folders
    - nnU-Net returns binary masks from argmax predictions (no thresholding needed)
        - Binary label maps are saved as TIFF files
        - Probability maps are saved separately via --save_probabilities flag
5) For DeepD3, performs thresholding to create binary masks for spines and dendrites
6) Post-processes predictions to extract spine and dendrite data
7) Computes spatial relationships (distance from root, distance from shaft, etc.)

This pipeline can be run standalone or as part of the full analysis workflow.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm
import tifffile
from itertools import chain
import shutil
from pathlib import Path

from spyne.core.spines.analysis.padding import pad_images, save_padded_images
from spyne.core.spines.analysis.spatial_distances import (
    euclidean_distance, find_closest_node, distance_along_neurite, find_root,
    prepare_neurite_distance_tools)
from spyne.core.spines.analysis.segmentation.post_processing import process_predictions
from spyne.core.spines.analysis.segmentation.post_processing_utils import threshold_prediction
from spyne.core.spines.analysis.backends.deepd3_backend import (
    DeepD3Backend, DeepD3Job)
from spyne.core.spines.analysis.backends.nnUnet_backend import (
    NNUNetBackend, NNUNetJob)


def run_inference_and_post_processing(
    images_folder: Path | str,
    spine_datasets: np.ndarray | list,
    original_dimensions: tuple | list,
    config: dict,
    probability_output_folder: Path | str | None = None,
    prediction_output_folder: Path | str | None = None,
) -> tuple:
    """
    Run the semantic segmentation pipeline: inference and post-processing.

    Parameters
    ----------
    images : np.ndarray | list
        Batch of ROI images for inference.
    spine_datasets : np.ndarray | list
        Batch of RoiSpine instances corresponding to the ROIs.
    original_dimensions : tuple | list
        Original dimensions of the images before padding.
    config : dict
        Configuration dictionary containing model path and parameters for segmentation.

        Expected keys:
        - device : str
            Type of device to use for computation.
        - segmentation_algorithm : str, optional
            Algorithm to use ('deepd3' or 'nnunet'). Default is 'deepd3'.
        - deepd3_model_fn : str
            Path to the DeepD3 model for segmentation.
        - nnunet_model_fn : str
            Path to the nnU-Net model for segmentation.
        - nnunet_python_executable : str, optional
            Path to Python executable in nnU-Net environment (required if using nnU-Net).
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
    probability_output_folder : Path | str | None, optional
        Folder to save probability maps (required for nnU-Net).
    prediction_output_folder : Path | str | None, optional
        Folder to save label predictions (required for nnU-Net and DeepD3).

    Returns
    -------
    tuple
        Processed predictions for spines and dendrites, and raw predictions.
    """
    # Get algorithm for progress bar description
    algorithm = config.get('segmentation_algorithm', 'deepd3').lower()

    with tqdm(
        total=2,
        desc=f"{algorithm.upper()} Inference",
        leave=True
        ) as pbar:


        # --------------------------------------------------------
        # Step 1: Perform inference based on algorithm choice
        # --------------------------------------------------------

        if algorithm == 'deepd3':
            
            # Use DeepD3Backend for isolated subprocess execution
            backend = DeepD3Backend(
                python_executable=config['segmentation']['deepd3_parameters']['deepd3_python_executable']
            )
            
            job = DeepD3Job(
                input_path=Path(images_folder),
                output_dir=Path(probability_output_folder),
                model_name=config['models']['deepd3_segmentation']['path'],
                original_dimensions=original_dimensions,
                device=config['segmentation']['deepd3_parameters']['device']
            )
            
            # Run inference and get results
            result = backend.run(job)
            
            # Check for errors
            if not result.success:
                raise RuntimeError(
                    f"DeepD3 inference failed:\n"
                    f"Command: {' '.join(result.command)}\n"
                    f"Return code: {result.return_code}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
            
            # Extract probability maps
            spine_probs = result.spine_probabilities
            dendrite_probs = result.dendrite_probabilities

            # Apply thresholding to create binary masks for post-processing
            dendrite_predictions = [
                threshold_prediction(
                    probs,
                    config['segmentation']['deepd3_parameters']['dendrite_threshold']
                    )
                for probs in dendrite_probs
            ]

            spine_predictions = [
                threshold_prediction(
                    probs,
                    config['segmentation']['deepd3_parameters']['spine_threshold']
                    )
                for probs in spine_probs
            ]

            for n in range(len(dendrite_predictions)):
                tifffile.imwrite(
                    Path(prediction_output_folder) / f"dendrite_prediction_{n:04}.tif",
                    dendrite_predictions[n].astype(np.uint8))
            
            for n in range(len(spine_predictions)):
                tifffile.imwrite(
                    Path(prediction_output_folder) / f"spine_prediction_{n:04}.tif",
                    spine_predictions[n].astype(np.uint8))
            
            # Store probability maps for .h5 saving
            raw_predictions = (spine_probs, dendrite_probs)

        elif algorithm == 'nnunet':
            
            # Use NNUNetBackend for isolated subprocess execution
            backend = NNUNetBackend(
                python_executable=config['segmentation']['nnunet_parameters']['nnunet_python_executable']
            )
            
            job = NNUNetJob(
                input_path=Path(images_folder),
                output_dir=Path(probability_output_folder),
                model_name=config['models']['nnunet_segmentation']['path'],
                original_dimensions=original_dimensions,
                dataset_id=config['segmentation']['nnunet_parameters'].get('dataset_id', '100'),
                dataset_name=config['segmentation']['nnunet_parameters'].get('dataset_name', 'SpineSegmentation'),
                configuration=config['segmentation']['nnunet_parameters'].get('configuration', '2d')
            )
            
            # Run inference and get results
            result = backend.run(job)
            
            # Check for errors
            if not result.success:
                raise RuntimeError(
                    f"nnU-Net inference failed:\n"
                    f"Command: {' '.join(result.command)}\n"
                    f"Return code: {result.return_code}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
            
            # Extract probability maps and binary masks
            spine_prob_maps = result.spine_probabilities
            dendrite_prob_maps = result.dendrite_probabilities
            spine_binary_masks = result.spine_binary_masks
            dendrite_binary_masks = result.dendrite_binary_masks
            
            # Convert to uint8 for post-processing (required by cv2.distanceTransform)
            spine_predictions = [mask.astype(np.uint8) for mask in spine_binary_masks]
            dendrite_predictions = [mask.astype(np.uint8) for mask in dendrite_binary_masks]
            
            # Save binary masks as TIFF
            for n in range(len(spine_predictions)):
                tifffile.imwrite(
                    Path(prediction_output_folder) / f"spine_prediction_{n:04}.tif",
                    spine_predictions[n].astype(np.uint8))
                tifffile.imwrite(
                    Path(prediction_output_folder) / f"dendrite_prediction_{n:04}.tif",
                    dendrite_predictions[n].astype(np.uint8))
            
            # Store probability maps for .h5 saving
            raw_predictions = (spine_prob_maps, dendrite_prob_maps)

        else:
            raise ValueError(
                f"Unknown segmentation algorithm: '{algorithm}'. "
                f"Supported algorithms: {config['segmentation']['supported_algorithms']}"
            )
        pbar.update(1)


        # --------------------------------------------------------
        # Step 2: Process predictions
        # --------------------------------------------------------

        # Merge algorithm-specific and shared post-processing parameters
        processing_config = config['segmentation']['post_processing_parameters']

        processed_predictions = process_predictions(
            spine_datasets=spine_datasets,
            spine_predictions=spine_predictions,
            dendrite_predictions=dendrite_predictions,
            config=processing_config
        )
        pbar.update(1)

    return processed_predictions, raw_predictions


def semantic_segmentation_pipeline(
        dataset,
        spine_dataset,
        config: dict,
        algorithm: str = None
) -> tuple:
    """
    Perform semantic segmentation pipeline for a dataset.

    This pipeline is self-contained and handles all necessary preparation:
    - Checks for cached denoised images (if denoising enabled in config)
    - Loads denoised images if cached, otherwise collects raw images
    - Pads images to uniform dimensions for batch processing
    - Runs segmentation inference
    - Post-processes predictions to extract spine and dendrite data

    Can be run standalone or as part of the full analysis workflow.

    Parameters
    ----------
    dataset : object
        The dataset containing ROIs for segmentation.
    spine_dataset : object
        The SpineDataset object to process each ROI.
    config : dict
        Configuration dictionary containing parameters
        for segmentation and inference.
    algorithm : str, optional
        Algorithm to use for segmentation ('deepd3' or 'nnunet').
        If not provided, it will be read from the config dictionary.

    Expected keys:
        - device : str
            Type of device to use for computation.
        - segmentation_algorithm : str, optional
            Algorithm to use ('deepd3' or 'nnunet'). Default is 'deepd3'.
        - segmentation_model_fn : str
            Path to the segmentation model.
        - nnunet_python_executable : str, optional
            Path to Python executable in nnU-Net environment (required if using nnU-Net).
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

    algorithm = algorithm if algorithm is not None else config.get(
        'segmentation', {}).get('segmentation_algorithm', 'deepd3').lower()
    
    if algorithm == 'deepd3':
        device = config['segmentation']['deepd3_parameters']['device']
        probability_output_folder = (
            spine_dataset.data_loader._files_mapping[
                'deepd3_probability_folder'])
        prediction_output_folder = (
            spine_dataset.data_loader._files_mapping[
                'deepd3_prediction_folder'])
    
    elif algorithm == 'nnunet':
        device = config['segmentation']['nnunet_parameters']['device']
        probability_output_folder = (
            spine_dataset.data_loader._files_mapping[
                'nnunet_probability_folder'])
        prediction_output_folder = (
            spine_dataset.data_loader._files_mapping[
                'nnunet_prediction_folder'])
    
    else:
        raise ValueError(
            f"Unknown segmentation algorithm: '{algorithm}'. "
            f"Supported algorithms: {config['segmentation']['supported_algorithms']}"
        )


    # --------------------------------------------------------
    # Step 1: Collect images and initialize RoiSpine instances
    # --------------------------------------------------------

    denoising_enabled = config.get('denoising', {}).get(
        'care_parameters', {}).get('enabled', False)
    
    # Check if denoised images exist (from previous denoising run)
    denoised_h5_path = (
        spine_dataset._dataset.folder / 
        spine_dataset.data_loader._files_mapping['care_denoised_images']
    )
    use_denoised = denoising_enabled and denoised_h5_path.exists()
    
    if use_denoised:
        print(f"Found cached denoised images at {denoised_h5_path}")
        print("Loading denoised images...")
        from spyne.core.spines.analysis.backends.csbdeep_backend import CSBDeepBackend
        
        # Load unpadded denoised images
        denoised_images = CSBDeepBackend.load_denoised_h5(denoised_h5_path)
        images = list(denoised_images)
        
        # Still need to initialize RoiSpine instances
        print("Initializing RoiSpine instances...")
        for roi_index in tqdm(
                range(len(dataset)),
                desc="Initializing spine datasets",
                total=len(dataset)):
            roi_spine = spine_dataset[roi_index]
            spine_datasets[roi_index] = roi_spine
    else:
        # Collect raw images from ROIs
        print("Collecting images from ROIs...")
        for roi_index in tqdm(
                range(len(dataset)),
                desc="Collecting images",
                total=len(dataset)):
            roi_spine = spine_dataset[roi_index]
            spine_datasets[roi_index] = roi_spine
            images[roi_index] = roi_spine.get_base_image()

    # --------------------------------------------------------
    # Step 2: Pad images for consistent batch dimensions
    # --------------------------------------------------------

    padded_images, original_dimensions = pad_images(images)
    padded_images = np.stack(padded_images, axis=0)
    
    # --------------------------------------------------------
    # Step 3: Prepare padded images folder
    # --------------------------------------------------------

    # Clean and prepare padded images folder
    padded_images_folder = Path(spine_dataset.data_loader._files_mapping['padded_images_folder']).resolve()
    
    # Remove existing files to avoid accumulation from previous runs
    if padded_images_folder.exists():
        shutil.rmtree(padded_images_folder)
    
    padded_images_folder.mkdir(parents=True, exist_ok=True)
    save_padded_images(padded_images, dataset.name, padded_images_folder)

    # Explicitly set algorithm in config for run_inference_and_post_processing
    config['segmentation_algorithm'] = algorithm


    # --------------------------------------------------------
    # Step 4: Run inference and post-processing
    # --------------------------------------------------------

    processed_predictions, raw_predictions = run_inference_and_post_processing(
        padded_images_folder,
        spine_datasets,
        original_dimensions,
        config,
        probability_output_folder=probability_output_folder,
        prediction_output_folder=prediction_output_folder
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

        if root_node_index is None:
            # Handle the case where no root is found
            # You could skip this spine, use a default, or raise an informative error
            raise ValueError(f"No root node found for node_id {closest_node_id}")
        
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

    # Convert dendrite predictions to boolean masks
    # Note: dendrite_predictions are already binary (uint8) at this point for both algorithms
    # - DeepD3: thresholded in run_inference_and_post_processing
    # - nnU-Net: argmax-based binary masks from inference
    # This conversion just creates boolean arrays (True/False) from the binary masks (0/1)
    dendrite_binary_mask = [
        pred > 0 for pred in dendrite_predictions
    ]

    # For spines, combine all spine masks per ROI via logical OR
    roi_masks: dict[int, np.ndarray] = {}
    for spine in spines_data:
        roi_n = spine['roi_n']
        if roi_n not in roi_masks:
            roi_masks[roi_n] = np.zeros_like(spine['mask'], dtype=bool)
        roi_masks[roi_n] |= spine['mask'] > 0

    # Create combined_spines_masks list, filling missing ROIs with zero arrays
    combined_spines_masks = []
    for roi_n, spine_pred in enumerate(spine_predictions):
        if roi_n in roi_masks:
            combined_spines_masks.append(roi_masks[roi_n])
        else:
            # No spines detected in this ROI, use zero array
            combined_spines_masks.append(
                np.zeros(spine_pred.shape, dtype=bool)
            )

    return (
        spine_datasets,
        spines_data,
        dendrites_data,
        spine_predictions,
        dendrite_predictions,
        dendrite_binary_mask,
        combined_spines_masks)