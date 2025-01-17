import torch
from spyne.core.zscore_classifier.model import ZScoreClassifier


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

    model = ZScoreClassifier()
    model.load_state_dict(torch.load(model_path))

    return model
