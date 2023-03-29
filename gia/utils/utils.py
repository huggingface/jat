import functools
import hashlib
import os
from typing import Callable

import numpy as np
import torch
from gym import spaces
from transformers.utils import logging

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


logger = logging.get_logger(__name__)
logger.setLevel("INFO")


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


def mu_law(x: np.ndarray, mu: float = 100, M: float = 256) -> np.ndarray:
    """
    μ-law companding.

    Args:
        x (np.ndarray): Input array
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.

    Returns:
        np.ndarray: Normalized array
    """
    return np.sign(x) * np.log(np.abs(x) * mu + 1.0) / np.log(M * mu + 1.0)


def discretize(x: np.ndarray, nb_bins: int = 1024) -> np.ndarray:
    """
    Discretize array.

    Example:
        >>> import numpy as np
        >>> x = np.array([-1.0, -0.1, 0.3, 0.4, 1.0])
        >>> discretize(x, nb_bins=6)
        array([0, 2, 3, 4, 5])

    Args:
        x (np.ndarray): Input array, in the range [-1, 1]
        nb_bins (int, optional): Number of bins. Defaults to 1024.

    Returns:
        np.ndarray: Discretized array
    """
    if np.any(x < -1.0) or np.any(x > 1.0):
        raise ValueError("Input array must be in the range [-1, 1]")
    x = (x + 1.0) / 2 * nb_bins  # [-1, 1] to [0, nb_bins]
    discretized = np.floor(x).astype(np.int64)
    discretized[discretized == nb_bins] = nb_bins - 1  # Handle the case where x == 1.0
    return discretized


def inverse_mu_law(x: np.ndarray, mu: float = 100, M: float = 256) -> np.ndarray:
    """
    Inverse μ-law companding.

    Args:
        x (np.ndarray): Input array
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.

    Returns:
        np.ndarray: Unnormalized array
    """
    return np.sign(x) * (np.exp(np.abs(x) * np.log(M * mu + 1.0)) - 1.0) / mu


def cache_decorator(func: Callable) -> Callable:
    """
    A decorator to cache the output of a function that loads torch objects. When the decorated function is called
    with the same parameters, the cached object is returned instead of calling the function again. The cache is
    stored as a file in the specified cache directory, with the filename derived from a hash of the function
    name and its arguments.

    This decorator is particularly useful for functions that load or generate large data objects, such as tensors
    or datasets, as it can help reduce the computational overhead and improve efficiency when the same objects
    are needed multiple times.

    Args:
        func (Callable): The function to be decorated. This function should return a torch object that can be
                         cached using torch.save and torch.load.

    Returns:
        Callable: The decorated function with caching behavior. The decorated function accepts two additional
            optional arguments:
            - load_from_cache_file (bool): If True, attempts to load the result from the cache file if it
                                            exists. If False, always calls the original function and saves
                                            the result to the cache file. Default: True.
            - cache_dir (str): The directory where cache files are stored. Default: "~/.cache/huggingface/datasets".

    Example:
        >>> import torch
        >>> @cache_decorator
        ... def dummy_func(x):
        ...     return x * x
        ...
        >>> dummy_func(torch.tensor([1, 2, 3]))
        tensor([1, 4, 9])
        >>> dummy_func(torch.tensor([1, 2, 3]))
        INFO:gia.utils.utils:Loading cache (/Users/quentingallouedec/.cache/huggingface/datasets/gia-
        f708447140b4ced19a1a27d9f28bc12254615c170d757cfe61f5925ec04b149d)
        tensor([1, 4, 9])
    """

    @functools.wraps(func)
    def wrapper(*args, load_from_cache_file: bool = True, cache_dir: str = "~/.cache/huggingface/datasets", **kwargs):
        # Get hash from the function parameters
        params = (func.__name__,) + args + tuple(kwargs.items())
        h = hashlib.sha256("".join(str(elem) for elem in params).encode()).hexdigest()
        cache_filename = f"gia-{h}"
        dirname = os.path.expanduser(cache_dir)
        os.makedirs(dirname, exist_ok=True)
        cache_path = os.path.join(dirname, cache_filename)

        if load_from_cache_file and os.path.exists(cache_path):
            logger.info(f"Loading cache ({cache_path})")
            return torch.load(cache_path)

        result = func(*args, **kwargs)

        # Save the result to cache
        torch.save(result, cache_path)

        return result

    return wrapper
