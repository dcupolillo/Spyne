import torch
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
