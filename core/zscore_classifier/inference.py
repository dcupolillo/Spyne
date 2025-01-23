from pathlib import Path
from tqdm import tqdm
import flammkuchen as fl
import torch
import numpy as np
from spyne.neuralnetwork.zscore_decoder.utils import set_device
from spyne.core.zscore_classifier.classifier import load_classifier


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
        A 1D numpy array representing the calcium trace (e.g., a single sweep or time series).
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
    - The model processes the input trace after it is reshaped to match the 
      expected input dimensions for the neural network (batch size, channels, and sequence length).
    - The model is set to evaluation mode (`model.eval()`) to ensure correct inference behavior.
    - The function returns a single probability score as output.
    - Ensure the input `sweep` is normalized or preprocessed to match the model's training conditions.
    """

    model.eval()

    sweep = torch.tensor(sweep, dtype=torch.float32)
    sweep = sweep.unsqueeze(0).unsqueeze(1)
    sweep = sweep.to(device)

    with torch.no_grad():
        output = model(sweep)
        # prediction = (output > 0.5).float().item()

    return output.item()


def detect_calcium_events(
        config: dict,
        batch_z_scores_BLA: list,
        batch_z_scores_CA3: list,
        output_folder: str or Path
) -> None:
    """
    Detect calcium events in spines using a trained neural network.

    Parameters
    ----------
    model_path : str, optional
        Path to the trained model for calcium event detection. Default is
        'zscore_best_model.pth'.

    Attributes Updated
    ------------------
    calcium_events_BLA : list
        Flags indicating calcium events for BLA spines.
    calcium_events_CA3 : list
        Flags indicating calcium events for CA3 spines.

    Notes
    -----
    - This method processes z-scores of spines using a neural network and
    identifies calcium events based on the model's predictions.
    - Results are saved as `.h5` files in the dataset's parent folder.

    Raises
    ------
    AssertionError
        If the length of event flags does not match the corresponding spine data.
    """
    
    model_path = config['classifier_model_fn']
    output_folder = Path(output_folder)

    calcium_events_BLA = []
    calcium_events_CA3 = []

    model = load_classifier(model_path)
    device = set_device()
    model.to(device)

    # Process BLA
    for spine in tqdm(batch_z_scores_BLA, desc="Analyzing BLA spines"):
        probabilities_BLA = []

        for sweep in spine:
            probabilities_BLA.append(
                is_calcium_event(sweep, model=model, device=device))
        calcium_events_BLA.append(probabilities_BLA)

    # Process CA3
    for spine in tqdm(batch_z_scores_CA3, desc="Analyzing CA3 spines"):
        probabilities_CA3 = []
        for sweep in spine:
            probabilities_CA3.append(
                is_calcium_event(sweep, model=model, device=device))
        calcium_events_CA3.append(probabilities_CA3)

    # Sanity check
    assert len(calcium_events_BLA) == len(batch_z_scores_BLA)
    for spine_probs, spine_traces in zip(
            calcium_events_BLA, batch_z_scores_BLA):
        assert len(spine_probs) == len(spine_traces)

    if calcium_events_BLA:
        fl.save(Path(output_folder, 'calcium_event_BLA.h5'), calcium_events_BLA)

    if calcium_events_CA3:
        fl.save(Path(output_folder, 'calcium_event_CA3.h5'), calcium_events_CA3)

    return calcium_events_BLA, calcium_events_CA3


def binarize_calcium_events_array(
        calcium_events: list,
        percentile: int
) -> tuple:
    """
    Binarize calcium event probabilities based on a dynamic threshold
    that depends on a percentile of the distribution of probabilities.

    This function takes a list of calcium event probabilities, computes a 
    dynamic threshold based on a specified percentile, and binarizes the 
    events such that probabilities above the threshold are set to 1, 
    otherwise 0.

    Parameters
    ----------
    calcium_events : list
        A nested list where each element corresponds to a spine, and each 
        spine contains a list of sweep probabilities.
    percentile : int
        The percentile value used to compute the dynamic threshold. For example, 
        a value of 99 will compute the 99th percentile.

    Returns
    -------
    tuple
        A tuple containing:
        - `dynamic_threshold` (float): The computed threshold value based on 
          the given percentile.
        - `binary_calcium_events` (list): A nested list with the same structure 
          as `calcium_events`, where probabilities above the threshold are set 
          to 1 and the rest to 0.

    Notes
    -----
    - NaN values in the `calcium_events` input are excluded when calculating 
      the percentile-based threshold.
    - The output retains the structure of the input list, making it easy to 
      trace binarized values back to their respective spines and sweeps.
    """

    probabilities = np.array(
        [sweep for spine in calcium_events for sweep in spine])
    probabilities = probabilities[~np.isnan(probabilities)]

    decision_boundary = np.percentile(probabilities, percentile)

    binary_calcium_events = np.zeros_like(
        calcium_events, dtype=int)

    for i, spine in enumerate(calcium_events):
        for n, sweep in enumerate(spine):
            if sweep >= decision_boundary:
                binary_calcium_events[i][n] = int(1)

    return decision_boundary, binary_calcium_events
