""" Created on Mon Oct 23 12:23:47 2023
    @author: dcupolillo """

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from spyne.core.spines.analysis.padding import unpad_predictions


def inference(
        images: np.ndarray or list,
        model_fn: str or Path,
        original_dimensions: tuple or list,
        device: str,
) -> tuple:
    """
    Perform neural network inference on an input batch of images to obtain pixel-wise
    segmentation predictions.

    This function handles both single 2D images and batches of 2D images. If the input
    is a single image, it is reshaped into a 3D array by adding a batch dimension.
    The predictions are generated using a pre-trained TensorFlow model, and the outputs
    are unpadded to restore their original dimensions.

    Parameters
    ----------
    images : np.ndarray or list
        The input images for segmentation. This can either be:
        - A single image (2D array).
        - A batch of images (3D array or a list of 2D arrays).
        Note: The images must already be padded to match the model's input size.
    model_fn : str or Path
        Path to the pre-trained TensorFlow model file.
    original_dimensions : tuple or list
        The original dimensions of the input images before padding. Used to restore
        the predictions to their original size.

    Returns
    -------
    tuple
        A tuple containing:
        - np.ndarray: Raw predictions for spines.
        - np.ndarray: Raw predictions for dendrites.

    Raises
    ------
    FileNotFoundError
        If the specified model file does not exist.

    Notes
    -----
    - Based on DeepD3 (https://github.com/ankilab/DeepD3)
    - The input images are normalized to the range [-1, 1] before being fed into the model.
    - The function attempts to use a GPU ('/gpu:0') for inference, falling back to the CPU
      if no GPU is available.
    - Predictions are automatically unpadded to match the original dimensions of the input images.
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
    with tf.device(device):
        model = load_model(model_path, compile=False)
        pd, ps = model.predict(images[..., None])

    ps = ps.squeeze()
    pd = pd.squeeze()

    spine_predictions = unpad_predictions(ps, original_dimensions)
    dendrite_predictions = unpad_predictions(pd, original_dimensions)

    if ps.ndim == 2:
        # Unpad images to original dimensions
        spine_predictions = unpad_predictions(ps, original_dimensions)
        dendrite_predictions = unpad_predictions(pd, original_dimensions)

    return spine_predictions, dendrite_predictions
