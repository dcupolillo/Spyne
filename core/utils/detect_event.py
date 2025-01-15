""" Created on Wed Sep 18 11:36:49 2024
    @author: dcupolillo """

import numpy as np
import torch


def is_calcium_event_threshold_crossing(
        sweep,
        threshold_factor=1.96,
        start=16,
        end=21,
        consecutive_points=2
) -> bool:
    """
    Threshold mothod to detect if a trace is an event or not

    Parameters
    ----------
    sweep : TYPE
        DESCRIPTION.
    threshold_factor : TYPE, optional
        DESCRIPTION. The default is 1.96.
    start : TYPE, optional
        DESCRIPTION. The default is 16.
    end : TYPE, optional
        DESCRIPTION. The default is 21.
    consecutive_points : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    bool
        DESCRIPTION.

    """

    std = np.std(sweep)
    threshold = threshold_factor * std

    for n in range(start, end - consecutive_points + 1):
        if all(sweep[n + i] > threshold for i in range(consecutive_points)):
            return True

    return False


def is_calcium_event(
        sweep,
        model,
        device
) -> bool:

    model.eval()

    sweep = torch.tensor(sweep, dtype=torch.float32)
    sweep = sweep.unsqueeze(0).unsqueeze(1)
    sweep = sweep.to(device)

    with torch.no_grad():
        output = model(sweep)
        # prediction = (output > 0.5).float().item()

    return output.item()


def binarize_calcium_events_array(
        calcium_events: list,
        percentile: int = 99):

    probabilities = np.array(
        [sweep for spine in calcium_events for sweep in spine])
    probabilities = probabilities[~np.isnan(probabilities)]
    dynamic_threshold = np.percentile(probabilities, percentile)

    binary_calcium_events = np.zeros_like(
        calcium_events, dtype=int)

    for i, spine in enumerate(calcium_events):
        for n, sweep in enumerate(spine):
            if sweep >= dynamic_threshold:
                binary_calcium_events[i][n] = int(1)

    return dynamic_threshold, binary_calcium_events
