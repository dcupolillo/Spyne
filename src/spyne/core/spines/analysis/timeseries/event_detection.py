from tqdm import tqdm
import numpy as np
import calcium_event_classifier as cec
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks


def detect_calcium_events(
        config: dict,
        zscores: np.ndarray,
        dFF: np.ndarray,
) -> np.ndarray:
    """
    Detect calcium events in spines using a trained neural network classifier.

    This function takes a numpy array of z-scored calcium traces and performs
    inference using a pre-trained neural network classifier to detect
    calcium events. The output is a numpy array of probabilities for each sweep
    in each spine.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters
        for the classifier model. Must include 'classifier_model_fn' key
        with the path to the trained model.
    zscores : np.ndarray
        A 2D numpy array where each row corresponds to a spine,
        and each column contains a z-scored calcium trace.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing calcium event probabilities for each spine.
        Shape: (n_spines, n_sweeps_per_spine). Each element is a probability
        for the corresponding sweep in that spine.
    """

    model_path = config['classifier_model_fn']

    # assert zscores.shape == dFF.shape, \
    #     "Z-scores and dFF traces must have the same shape."

    model = cec.load_classifier_dff(model_path)
    
    device = cec.set_device()
    model.to(device)

    calcium_probabilities = np.empty(
        [zscores.shape[0], zscores.shape[1]], dtype=np.float32)
    logits = np.empty(
        [zscores.shape[0], zscores.shape[1]], dtype=np.float32)

    for n_spine, spine in enumerate(tqdm(zscores, desc="Analyzing spines")):
        for n_sweep, sweep in enumerate(spine):
            (
                logits[n_spine][n_sweep],
                calcium_probabilities[n_spine][n_sweep]
            ) = cec.is_calcium_event(
                zscore=sweep, dFF=dFF[n_spine][n_sweep],
                model=model, device=device)

    return calcium_probabilities


def find_threshold(
        probabilities: np.ndarray,
        bandwidth: float = 0.02,
        min_distance: int = 50,
        prominence: float = 0.01,
) -> float:
    """
    Find the threshold for calcium event detection by identifying
    minima between modes.

    This function uses Kernel Density Estimation (KDE)to estimate the
    probability distribution and then finds the minimum between modes
    as the threshold. This approach automatically identifies the natural
    separation point between different populations
    (e.g., non-events vs events).

    Parameters
    ----------
    probabilities : np.ndarray
        A 2D numpy array of calcium event probabilities with shape
        (n_spines, n_sweeps_per_spine).
    bandwidth : float, optional
        Bandwidth for the KDE. Default is 0.02.
    min_distance : int, optional
        Minimum distance between peaks in the KDE for peak detection.
        Default is 50 (out of 1000 points).
    prominence : float, optional
        Minimum prominence of peaks for detection. Default is 0.01.

    Returns
    -------
    float
        The threshold value for calcium event detection
        (minimum between modes).

    Notes
    -----
    - The function uses Kernel Density Estimation to estimate the distribution
        of calcium event probabilities.
    - Peaks (modes) are detected in the probability density function.
        Assumes bimodal distribution.
    - The threshold is set at the minimum between the two most prominent modes.
    - If only one mode is detected, returns the median of the probabilities.
    - The output is a single float value representing the threshold.
    """

    # Flatten the probabilities and remove NaN values
    probabilities = probabilities.flatten()
    probabilities = probabilities[~np.isnan(probabilities)]

    if len(probabilities) == 0:
        raise ValueError("No valid calcium event probabilities provided.")

    probs_reshaped = probabilities[:, np.newaxis]

    kde = KernelDensity(
        kernel='gaussian', bandwidth=bandwidth).fit(probs_reshaped)

    # Evaluate KDE on a grid of points between 0 and 1
    x_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)

    # Find peaks (modes) in the density
    peaks, properties = find_peaks(
        dens,
        distance=min_distance,
        prominence=prominence
    )

    if len(peaks) < 2:
        # If less than 2 peaks found, fall back to median
        print(f"Warning: Only {len(peaks)} mode(s) detected. "
              "Using median as threshold.")
        return np.median(probabilities)

    # Sort peaks by prominence (highest first)
    peak_prominences = properties['prominences']
    sorted_indices = np.argsort(peak_prominences)[::-1]
    top_peaks = peaks[sorted_indices[:2]]  # Get top 2 peaks

    # Find the minimum between the two most prominent peaks
    left_peak, right_peak = sorted(top_peaks)

    # Find minimum in the region between peaks
    region_start = left_peak
    region_end = right_peak
    region_dens = dens[region_start:region_end+1]

    if len(region_dens) == 0:
        return np.median(probabilities)

    min_idx_in_region = np.argmin(region_dens)
    min_idx_global = region_start + min_idx_in_region

    threshold = x_plot[min_idx_global, 0]

    return threshold


def binarize_calcium_event_probabilities(
        calcium_event_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Binarize calcium event probabilities based on a threshold.

    This function takes a numpy array of calcium event probabilities and
    binarizes the events such that probabilities above the threshold
    are set to 1, otherwise 0. The threshold is automatically loaded
    from the model checkpoint.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters
        for the classifier model. Must include 'classifier_model_fn' key
        with the path to the trained model containing the threshold.
    calcium_event_probabilities : np.ndarray
        A 2D numpy array of calcium event probabilities with shape
        (n_spines, n_sweeps_per_spine).

    Returns
    -------
    np.ndarray
        A 2D numpy array with the same structure as
        `calcium_event_probabilities`, where probabilities above
        the threshold are set to 1 and the rest to 0.
        Data type is int.

    Notes
    -----
    - The threshold is automatically extracted from the model's checkpoint
      using the key 'best_thresholds'[-1].
    - NaN values in the `calcium_event_probabilities` input are excluded when
      performing the comparison.
    - The output retains the structure of the input array,
      making it easy to trace binarized values back to
      their respective spines and sweeps.
    """

    threshold = find_threshold(calcium_event_probabilities)

    probabilities_flat = calcium_event_probabilities.flatten()
    probabilities_flat = probabilities_flat[~np.isnan(probabilities_flat)]

    calcium_events_binary = np.zeros_like(
        calcium_event_probabilities, dtype=int)

    for n_spine, spine in enumerate(calcium_event_probabilities):
        for n_sweep, sweep in enumerate(spine):
            if sweep >= threshold:
                calcium_events_binary[n_spine][n_sweep] = int(1)

    return calcium_events_binary
