import numpy as np
import torch
from gym import spaces
from torch import Tensor

# numpy_to_torch_dtype_dict = {
#     np.bool       : torch.bool,
#     np.uint8      : torch.uint8,
#     np.int8       : torch.int8,
#     np.int16      : torch.int16,
#     np.int32      : torch.int32,
#     np.int64      : torch.int64,
#     np.float16    : torch.float16,
#     np.float32    : torch.float32,
#     np.float64    : torch.float64,
#     np.complex64  : torch.complex64,
#     np.complex128 : torch.complex128
# }
# torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def check_space_is_flat_dict(space: spaces.Dict):
    for k, v in space.items():
        assert isinstance(v, (spaces.Box, spaces.Discrete)), "An instance the thie space {space} is not flat {v}"


def _call_remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def lod_to_dol(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}


def dol_to_lod(dol):
    return [dict(zip(dol, t)) for t in zip(*dol.values())]


def dol_to_donp(dol):
    return {k: np.array(v) for k, v in dol.items()}


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


def inverse_mu_law(x: Tensor, mu: float = 100, M: float = 256) -> Tensor:
    """
    Inverse μ-law companding.

    Args:
        x (Tensor): Input tensor
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.

    Returns:
        Tensor: Unnormalized tensor
    """
    return torch.sign(x) * (torch.exp(torch.abs(x) * torch.log(torch.tensor(M * mu + 1.0))) - 1.0) / mu


def to_channel_first(image_tensor):
    """
    Convert a tensor from channel-last to channel-first format (used by PyTorch).

    Args:
        image_tensor (Tensor): Batch of images.

    Returns:
        Tensor: Batch of images in channel-first format.

    Raises:
        ValueError: If the input tensor is not detected to be an image.

    Example:
        >>> x = torch.rand(4, 84, 84, 3)
        >>> to_channel_first(x).shape
        torch.Size([4, 3, 84, 84])
    """
    # Get the index of the smallest dimension, expect the batch dimension
    channel_dim = torch.argmin(torch.tensor(image_tensor.shape[1:])) + 1

    # Check if the input is in channel-first or channel-last format
    if channel_dim == 1:
        # The tensor is already in channel-first format
        return image_tensor
    elif channel_dim == 3:
        # The tensor is in channel-last format, permute the tensor
        return image_tensor.permute(0, 3, 1, 2)
    else:
        # Invalid tensor shape
        raise ValueError("Invalid tensor shape: {}".format(image_tensor.shape))
