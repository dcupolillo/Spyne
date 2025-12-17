""" Created on Mon Oct 30 12:27:04 2023
    @author: dcupolillo """

import numpy as np
import cv2


def median(
        image: np.ndarray,
        kernel: int
) -> np.ndarray:
    """
    Image filtering.

    Parameters
    ----------
    image : np.ndarray
        The input image to be filtered.
    kernel : int
        Kernel size (must be odd).

    Raises
    ------
    ValueError
        Kernel size must be odd.

    Returns
    -------
    np.ndarray
        Filtered image.

    """

    if kernel % 2 == 0:
        raise ValueError("Kernel size must be odd")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return cv2.medianBlur(image, kernel)


def gaussian(
        image: np.ndarray,
        kernel: int,
        sigma: float
) -> np.ndarray:
    """
    Gaussian filter implementation.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    kernel : int
        Kernel size.
    sigma : float
        Gaussian kernel standard deviation.

    Returns
    -------
    np.ndarray
        The blurred image.

    """
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return cv2.GaussianBlur(image, (kernel, kernel), sigma)
