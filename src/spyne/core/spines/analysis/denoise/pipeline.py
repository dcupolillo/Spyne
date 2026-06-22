"""
Module to perform CARE denoising using CSBDeep backend.

This module provides a modular denoising pipeline that separates workflow
logic from orchestration, similar to the segmentation pipeline structure.

The main function is `denoising_pipeline()`, which handles:
1) Image collection and padding
2) Backend execution (CSBDeep CARE) - unpadding handled internally
3) Saving results (TIFF and h5)

Structure mirrors the segmentation pipeline:
- Backend handles inference AND unpadding
- Pipeline handles I/O operations

Created on January 21, 2026
@author: dcupolillo
"""

from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np

from spyne.core.spines.analysis.padding import pad_images, save_padded_images
from spyne.core.spines.analysis.backends.csbdeep_backend import (
    CSBDeepBackend, CSBDeepJob)


def run_denoising_and_post_processing(
    input_folder: Path,
    output_folder: Path,
    original_dimensions: tuple | list,
    config: dict,
    save_h5: bool = True,
    h5_output_path: Path | None = None
) -> np.ndarray:
    """
    Run CARE denoising and post-processing (saving only).

    Backend handles unpadding internally for consistency with
    segmentation backends.

    Parameters
    ----------
    input_folder : Path
        Directory containing padded input images for CARE.
    output_folder : Path
        Directory where CARE will save denoised TIFF outputs.
    original_dimensions : tuple | list
        Original dimensions of images before padding.
    config : dict
        Configuration dictionary containing CARE parameters.
        Expected keys:
        - care_model_fn : str or Path
            Path to the CARE model.
        - care_axes : str
            Image axes (e.g., "YX" for 2D).
        - care_python_executable : str or Path
            Path to Python executable in CSBDeep environment.
    save_h5 : bool, optional
        Whether to save denoised images as h5 file. Default is True.
    h5_output_path : Path, optional
        Path for h5 output file. Required if save_h5=True.

    Returns
    -------
    np.ndarray
        Denoised images with original (unpadded) dimensions.

    Raises
    ------
    RuntimeError
        If CARE denoising fails.
    """
    # Setup CSBDeep job with original dimensions for unpadding
    care_job = CSBDeepJob(
        input_path=input_folder,
        output_dir=output_folder,
        model_name=config['care_model_fn'],
        axes=config['care_axes'],
        original_dimensions=original_dimensions
    )

    # Run CARE denoising (backend handles unpadding)
    try:
        print("Running CARE denoising...")
        care_backend = CSBDeepBackend(
            python_executable=config['care_python_executable']
        )
        care_result = care_backend.run(care_job)

        # Backend returns unpadded images
        denoised_images = care_result.denoised_images

        # Save as h5 file if requested
        if save_h5:
            if h5_output_path is None:
                h5_output_path = output_folder.parent / 'care_denoised_images.h5'
            print(f"Saving denoised images as h5 to {h5_output_path}...")
            CSBDeepBackend.save_as_h5(denoised_images, h5_output_path)

        # Clean up temporary folders
        for tmp_folder in (output_folder, input_folder):
            if Path(tmp_folder).exists():
                shutil.rmtree(tmp_folder)

        print(f"Denoising complete. Processed {len(denoised_images)} images.")
        return denoised_images

    except Exception as e:
        raise RuntimeError(f"CARE denoising failed: {e}") from e


def denoising_pipeline(
    spine_dataset,
    dataset,
    config: dict,
    input_folder: Path,
    output_folder: Path,
    save_h5: bool = True,
    h5_output_path: Path | None = None
) -> np.ndarray:
    """
    Run CARE denoising pipeline.

    This function orchestrates the complete denoising workflow:
    - Collects images from ROIs
    - Pads images to uniform dimensions
    - Runs CSBDeep CARE backend
    - Unpads denoised results to original dimensions

    Parameters
    ----------
    spine_dataset : SpineDataset
        The SpineDataset instance to collect images from.
    dataset : ImagingDataset
        The imaging dataset (for accessing metadata if needed).
    config : dict
        Configuration dictionary containing CARE parameters.
        Expected keys:
        - care_model_fn : str or Path (path to CARE model)
        - care_axes : str (image axes e.g., "YX" for 2D)
        - care_python_executable : str or Path (Python in CSBDeep env)
    dataset_name : str
        Name for output files.
    input_folder : Path
        Directory to save temporary input images for CARE.
    output_folder : Path
        Directory where CARE will save denoised outputs.
    save_h5 : bool, optional
        Whether to save denoised images as h5 file. Default is True.
    h5_output_path : Path, optional
        Path for h5 output file. Required if save_h5=True.

    Returns
    -------
    np.ndarray
        Denoised images with original (unpadded) dimensions.

    Raises
    ------
    RuntimeError
        If CARE denoising fails.

    Notes
    -----
    This function handles the complete denoising workflow including image
    collection and padding. Does not manage file existence checks - that's
    handled by the orchestration layer.
    """
    # Collect images from ROIs
    print("Collecting images from ROIs for denoising...")
    images = [spine_dataset[i].get_base_image() for i in range(len(dataset))]

    # --------------------------------------------------------
    # Step 1: Pad images for consistent batch dimensions
    # --------------------------------------------------------
    
    print("Padding images for batch denoising...")
    padded_images, original_dimensions = pad_images(images)
    padded_images = np.stack(padded_images, axis=0)

    # --------------------------------------------------------
    # Step 2: Prepare padded images folder
    # --------------------------------------------------------
    
    # Clean and prepare padded images folder
    padded_images_folder = Path(spine_dataset.data_loader._files_mapping['padded_images_folder']).resolve()
    if padded_images_folder.exists():
        shutil.rmtree(padded_images_folder)
    
    padded_images_folder.mkdir(parents=True, exist_ok=True)
    save_padded_images(padded_images, dataset.name, padded_images_folder)

    # --------------------------------------------------------
    # Step 3: Run denoising and post-processing
    # --------------------------------------------------------

    denoised_images = run_denoising_and_post_processing(
        input_folder=input_folder,
        output_folder=output_folder,
        original_dimensions=original_dimensions,
        config=config,
        save_h5=save_h5,
        h5_output_path=h5_output_path
    )

    return denoised_images
