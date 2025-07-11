from pathlib import Path
from tqdm import tqdm
import flammkuchen as fl
import torch
import numpy as np

import zscore_classifier as zsc
# from spyne.neuralnetwork.zscore_decoder.utils import set_device
from spyne.core.timeseries.classifier import load_classifier


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
        zscores_BLA: list,
        zscores_CA3: list,
        save: bool,
        output_folder: str or Path
) -> None:
    """
    Detect calcium events in spines using a trained neural network classifier.

    This function takes a list of z-scored calcium traces from
    BLA and CA3 spines, and performs inference using a pre-trained
    neural network classifier to detect calcium events.
    The output is a list of probabilities for each sweep
    in each spine.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters
        for the classifier model.
    zscores_BLA : list
        A nested list where each element corresponds to a spine,
        and each spine contains
        a list of z-scored calcium traces from BLA.
    zscores_CA3 : list
        A nested list where each element corresponds to a spine,
        and each spine contains
        a list of z-scored calcium traces from CA3.
    save : bool, optional
        Whether to save the calcium event probabilities to disk.
    output_folder : str or Path
        The folder where the calcium event probabilities will be saved.

    Returns
    -------
    tuple
        A tuple containing the calcium event probabilities for BLA
        and CA3 spines.
    """

    model_path = config['classifier_model_fn']
    output_folder = Path(output_folder)

    calcium_events_BLA = []
    calcium_events_CA3 = []

    model = load_classifier(model_path)
    device = zsc.set_device()
    model.to(device)

    # Process BLA
    for spine in tqdm(zscores_BLA, desc="Analyzing BLA spines"):
        probabilities_BLA = []

        for sweep in spine:
            probabilities_BLA.append(
                is_calcium_event(sweep, model=model, device=device))
        calcium_events_BLA.append(probabilities_BLA)

    # Process CA3
    for spine in tqdm(zscores_CA3, desc="Analyzing CA3 spines"):
        probabilities_CA3 = []
        for sweep in spine:
            probabilities_CA3.append(
                is_calcium_event(sweep, model=model, device=device))
        calcium_events_CA3.append(probabilities_CA3)

    # Sanity check
    assert len(calcium_events_BLA) == len(zscores_BLA)
    for spine_probs, spine_traces in zip(
            calcium_events_BLA, zscores_BLA):
        assert len(spine_probs) == len(spine_traces)

    if save and output_folder:
        if calcium_events_BLA:
            fl.save(
                Path(output_folder, 'calcium_events_BLA.h5'),
                calcium_events_BLA)

        if calcium_events_CA3:
            fl.save(
                Path(output_folder, 'calcium_events_CA3.h5'),
                calcium_events_CA3)

    return calcium_events_BLA, calcium_events_CA3


def binarize_calcium_events_array(
        calcium_events_BLA: list,
        calcium_events_CA3: list,
        percentile: int,
        save: bool,
        output_folder: str or Path
) -> tuple:
    """
    Binarize calcium event probabilities for BLA and CA3 events based on
    a percentile-based dynamic threshold of the distribution of probabilities.

    This function takes a list of calcium event probabilities, computes a 
    dynamic threshold based on a specified percentile, and binarizes the 
    events such that probabilities above the threshold are set to 1, 
    otherwise 0.

    Parameters
    ----------
    calcium_events_BLA : list
        A nested list where each element corresponds to a spine, and each 
        spine contains a list of probabilities of having a BLA event.
    calcium_events_CA3 : list
        A nested list where each element corresponds to a spine, and each
        spine contains a list of probabilities of having a CA3 event.
    percentile : int
        The percentile value used to compute the dynamic threshold. For example, 
        a value of 99 will compute the 99th percentile.
    save : bool, optional
        Whether to save the binarized calcium events to disk.
    output_folder : str or Path
        The folder where the binarized calcium events will be saved.

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

    probabilities_BLA = np.array(
        [sweep for spine in calcium_events_BLA for sweep in spine])
    probabilities_BLA = probabilities_BLA[~np.isnan(probabilities_BLA)]

    probabilities_CA3 = np.array(
        [sweep for spine in calcium_events_CA3 for sweep in spine])
    probabilities_CA3 = probabilities_CA3[~np.isnan(probabilities_CA3)]

    decision_boundary_BLA = np.percentile(probabilities_BLA, percentile)
    decision_boundary_CA3 = np.percentile(probabilities_CA3, percentile)

    calcium_events_binary_BLA = np.zeros_like(
        calcium_events_BLA, dtype=int)
    
    calcium_events_binary_CA3 = np.zeros_like(
        calcium_events_CA3, dtype=int)

    for i, spine in enumerate(calcium_events_BLA):
        for n, sweep in enumerate(spine):
            if sweep >= decision_boundary_BLA:
                calcium_events_binary_BLA[i][n] = int(1)

    for i, spine in enumerate(calcium_events_CA3):
        for n, sweep in enumerate(spine):
            if sweep >= decision_boundary_CA3:
                calcium_events_binary_CA3[i][n] = int(1)

    if save and output_folder:
        if calcium_events_binary_BLA is not None:
            fl.save(
                Path(output_folder, 'calcium_events_binary_BLA.h5'),
                calcium_events_binary_BLA)

        if calcium_events_binary_CA3 is not None:
            fl.save(
                Path(output_folder, 'calcium_events_binary_CA3.h5'),
                calcium_events_binary_CA3)

    return (
        decision_boundary_BLA,
        calcium_events_binary_BLA,
        decision_boundary_CA3,
        calcium_events_binary_CA3
    )
