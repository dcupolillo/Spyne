"""
Functions to handle padding and unpadding of images for batch processing
of the various analysis backends. Ensures that images are padded to the required
dimensions for neural network models and that predictions are unpadded to restore
original dimensions. Also includes utility functions for saving padded images to disk.
"""


from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile


def pad_image(
        image: np.ndarray,
        target_height: int,
        target_width: int,
) -> np.ndarray:
    """
    Pad a single image to the target height and width.
    Instrumental to batch processing, where images
    should have the same size.

    Building block to pad_images().

    Parameters
    ----------
    image : np.ndarray
        A 2D array representing the image to be padded.
    target_height : int
        The desired height of the padded image.
    target_width : int
        The desired width of the padded image.

    Returns
    -------
    np.ndarray
        The padded image with the specified dimensions.

    Notes
    -----
    - The padding is performed symmetrically using the 'reflect' mode.
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
        images: np.ndarray | list,
        pad_size: int = 32,
) -> tuple[np.ndarray | list, tuple | list]:
    """
    Pad a single image or a list of images to meet the size requirements 
    for the neural network model. Ensures dimensions are multiples of the 
    specified padding size.

    Parameters
    ----------
    images : np.ndarray or list
        The input image(s). Can be:
        - A single image (2D array).
        - A list of images (each a 2D array).
    pad_size : int, optional
        The padding size. Ensures the final dimensions are multiples of 
        this value (default is 32).

    Returns
    -------
    tuple
        - np.ndarray or list: The padded image(s).
        - tuple or list: The original dimensions of the image(s).

    Notes
    -----
    - If a single image is provided, it returns the padded image and its 
      original dimensions as a tuple.
    - For a list of images, it returns a list of padded images and a 
      corresponding list of original dimensions.
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
        prediction: np.ndarray,
        original_dimensions: tuple,
) -> np.ndarray:
    """
    Remove padding from a single prediction to restore its original dimensions.

    Building block to unpad_predictions().

    Parameters
    ----------
    prediction : np.ndarray
        The padded prediction image (2D array).
    original_dimensions : tuple
        A tuple (height, width) specifying the original dimensions
        before padding.

    Returns
    -------
    np.ndarray
        The unpadded prediction image.

    Notes
    -----
    - The function assumes the input dimensions are larger than or equal
      to the original dimensions.
    """

    original_height, original_width = original_dimensions

    return prediction[:original_height, :original_width]


def unpad_predictions(
        predictions: list[np.ndarray] | np.ndarray,
        original_dimensions: tuple[int, ...] | list[tuple[int, ...]],
) -> np.ndarray | list[np.ndarray]:
    """
    Remove padding from a list of predictions or a single prediction to 
    restore their original dimensions.

    Parameters
    ----------
    predictions : list or np.ndarray
        The input predictions. Can be:
        - A single prediction (2D array).
        - A list of predictions (each a 2D array).
    original_dimensions : tuple or list
        The original dimensions of the input predictions before padding. Can be:
        - A tuple for a single prediction.
        - A list of tuples for a batch of predictions.

    Returns
    -------
    np.ndarray or list
        The unpadded predictions. If a single prediction is provided, 
        it returns a 2D array. If a batch of predictions is provided, 
        it returns a list of 2D arrays.

    Notes
    -----
    - The function automatically determines whether the input is a batch 
      of predictions or a single prediction based on its dimensions.
    - For a batch, the predictions are unpadded individually based on 
      their corresponding original dimensions.
    """

    if predictions.ndim != 2:
        unpadded_predictions = [
            unpad_prediction(pred, dims)
            for pred, dims in zip(predictions, original_dimensions)]

    else:
        unpadded_predictions = unpad_prediction(
            predictions, original_dimensions)

    return unpadded_predictions


def save_padded_images(
    padded_images: np.ndarray,
    dataset_name: str,
    output_folder: Path
) -> None:
    """
    Save padded images to disk as TIFF files.
    Saves each image with a filename pattern: {dataset_name}_{index:04d}_0000.tif.
    The final _0000 suffix indicates channel 0 (required for nnUNet compatibility).
    The {index:04d} is the enumeration that identifies each ROI.
    This is a preparatory step for batch inference.

    Parameters
    ----------
    padded_images : np.ndarray
        Array of padded images to save.
    dataset_name : str
        Name of the dataset, used for naming the output files.
    output_folder : Path
        Directory where the padded images will be saved.

    Returns
    -------
    None
        This function saves the images to disk and does not return anything.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, padded_image in enumerate(padded_images):
        output_path = output_folder / f"{dataset_name}_{i:04d}_0000.tif"
        tifffile.imwrite(output_path, padded_image)
