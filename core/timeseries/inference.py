from pathlib import Path
from tqdm import tqdm
import flammkuchen as fl
import torch
import numpy as np
import zscore_classifier as zsc


def load_classifier(model_path: str) -> torch.nn.Module:
    """
    Load a pre-trained Z-Score neural network classifier.

    Parameters
    ----------
    model_path : str
        Path to the saved PyTorch model file (.pth).

    Returns
    -------
    torch.nn.Module
        The loaded ZScoreNN model in evaluation mode.
    """

    model = zsc.ZScoreClassifier()
    model.load_state_dict(torch.load(model_path))

    return model


def is_calcium_event(
        sweep: np.ndarray,
        model: torch.nn.Module,
        device: str,
) -> bool:
    """
    Perform inference on a single calcium trace to determine event probability.

    Parameters
    ----------
    sweep : np.ndarray
        A 1D numpy array representing the calcium trace
        (e.g., a single sweep or time series).
    model : torch.nn.Module
        A trained PyTorch neural network model for calcium event detection.
    device : str
        The device to perform inference on ('cpu' or 'cuda').

    Returns
    -------
    float
        The probability of a calcium event as predicted by the model.

    Notes
    -----
    - The model processes the input trace after it is reshaped
        to match the expected input dimensions for the
        neural network (batch size, channels, and sequence length).
    - The model is set to evaluation mode (`model.eval()`)
        to ensure correct inference behavior.
    - The function returns a single probability score as output.
    - Ensure the input `sweep` is normalized or preprocessed
        to match the model's training conditions.
    """

    model.eval()

    sweep = torch.tensor(sweep, dtype=torch.float32)
    sweep = sweep.unsqueeze(0).unsqueeze(1)
    sweep = sweep.to(device)

    with torch.no_grad():
        output = model(sweep)

    return output.item()


def detect_calcium_events(
        config: dict,
        zscores: list,
) -> list:
    """
    Detect calcium events in spines using a trained neural network classifier.

    This function takes a list of z-scored calcium traces and performs 
    inference using a pre-trained neural network classifier to detect 
    calcium events. The output is a list of probabilities for each sweep
    in each spine.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters
        for the classifier model. Must include 'classifier_model_fn' key
        with the path to the trained model.
    zscores : list
        A nested list where each element corresponds to a spine,
        and each spine contains a list of z-scored calcium traces.

    Returns
    -------
    list
        A nested list containing calcium event probabilities for each spine.
        Each element corresponds to a spine, containing a list of 
        probabilities for each sweep in that spine.
    """

    model_path = config['classifier_model_fn']
    output_folder = Path(output_folder)

    calcium_probabilities = []

    model = load_classifier(model_path)
    device = zsc.set_device()
    model.to(device)

    for spine in tqdm(zscores, desc="Analyzing spines"):
        probabilities = []

        for sweep in spine:
            probabilities.append(
                is_calcium_event(sweep, model=model, device=device))
        calcium_probabilities.append(probabilities)

    # Sanity check
    assert len(calcium_probabilities) == len(zscores)
    assert all(len(spine_probs) == len(spine_traces) for spine_probs, spine_traces in zip(
        calcium_probabilities, zscores))    

    return calcium_probabilities


def binarize_calcium_event_probabilities(
        config: dict,
        calcium_event_probabilities: list,
) -> np.ndarray:
    """
    Binarize calcium event probabilities based on a threshold.

    This function takes a list of calcium event probabilities and
    binarizes the events such that probabilities above the threshold 
    are set to 1, otherwise 0.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters
        for the classifier model. Must include 'classifier_cutoff' key.
    threshold : float
        The threshold value used for binarization. Probabilities above
        this value will be set to 1, others to 0.

    Returns
    -------
    np.ndarray
        A nested array with the same structure as `calcium_probabilities`, 
        where probabilities above the threshold are set to 1 and the rest to 0.

    Notes
    -----
    - NaN values in the `calcium_probabilities` input are excluded when
        performing the comparison.
    - The output retains the structure of the input list,
        making it easy to trace binarized values back to
        their respective spines and sweeps.
    """

    threshold = config['classifier_cutoff']

    probabilities_flat = np.array(
        [sweep for spine in calcium_event_probabilities for sweep in spine])
    probabilities_flat = probabilities_flat[~np.isnan(probabilities_flat)]

    calcium_events_binary = np.zeros_like(
        calcium_event_probabilities, dtype=int)

    for i, spine in enumerate(calcium_event_probabilities):
        for n, sweep in enumerate(spine):
            if sweep >= threshold:
                calcium_events_binary[i][n] = int(1)

    return calcium_events_binary
