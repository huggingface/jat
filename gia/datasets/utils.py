import torch
from torch import Tensor
import numpy as np


def mu_law(x: Tensor, mu: float = 100, M: float = 256) -> Tensor:
    """
    μ-law companding.
    Args:
        x (Tensor): Input tensor
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.
    Returns:
        Tensor: Normalized tensor
    """
    return torch.sign(x) * torch.log(torch.abs(x) * mu + 1.0) / torch.log(torch.tensor(M * mu + 1.0))


def discretize(x: Tensor, nb_bins: int = 1024) -> Tensor:
    """
    Discretize tensor.
    Example:
        >>> x = tensor([-1.0, -0.1, 0.3, 0.4, 1.0])
        >>> discretize(x, nb_bins=6)
        tensor([0, 2, 3, 4, 5])
    Args:
        x (Tensor): Input tensor, in the range [-1, 1]
        nb_bins (int, optional): Number of bins. Defaults to 1024.
    Returns:
        Tensor: Discretized tensor
    """
    if torch.any(x < -1.0) or torch.any(x > 1.0):
        raise ValueError("Input tensor must be in the range [-1, 1]")
    x = (x + 1.0) / 2 * nb_bins  # [-1, 1] to [0, nb_bins]
    discretized = torch.floor(x).long()
    discretized[discretized == nb_bins] = nb_bins - 1  # Handle the case where x == 1.0
    return discretized


def mu_law_np(x: np.ndarray, mu: float = 100, M: float = 256) -> Tensor:
    """
    μ-law companding.
    Args:
        x (np.Array): Input numpy array
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.
    Returns:
        np.Array: Normalized tensor
    """
    return np.sign(x) * np.log(np.abs(x) * mu + 1.0) / np.log(M * mu + 1.0)


def discretize_np(x: np.ndarray, nb_bins: int = 1024) -> Tensor:
    """
    Discretize tensor.
    Example:
        >>> x = tensor([-1.0, -0.1, 0.3, 0.4, 1.0])
        >>> discretize(x, nb_bins=6)
        tensor([0, 2, 3, 4, 5])
    Args:
        x (Tensor): Input tensor, in the range [-1, 1]
        nb_bins (int, optional): Number of bins. Defaults to 1024.
    Returns:
        Tensor: Discretized tensor
    """
    if np.any(x < -1.0) or np.any(x > 1.0):
        raise ValueError("Input tensor must be in the range [-1, 1]")
    x = (x + 1.0) / 2 * nb_bins  # [-1, 1] to [0, nb_bins]
    discretized = np.floor(x).astype(np.uint32)
    discretized[discretized == nb_bins] = nb_bins - 1  # Handle the case where x == 1.0
    return discretized


def tokenize_np(x: np.ndarray, mu: float = 100, M: float = 256, nb_bins: int = 1024):
    x = mu_law_np(x)
    x = np.clip(x, -1.0, 1.0)
    return discretize_np(x)
