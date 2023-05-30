import functools
import hashlib
import inspect
import os
from typing import Callable

import torch
from transformers.utils import logging
from uncertainties import ufloat
from uncertainties.core import Variable

logger = logging.get_logger(__name__)
logger.setLevel("INFO")


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
            - load_from_cache (bool): If True, attempts to load the result from the cache file if it
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
    def wrapper(*args, load_from_cache: bool = True, cache_dir: str = "~/.cache/huggingface/datasets", **kwargs):
        if "load_from_cache" in inspect.getfullargspec(func).args:
            # add it if the inner function allow this kwarg
            kwargs["load_from_cache"] = load_from_cache
        if "cache_dir" in inspect.getfullargspec(func).args:
            # add it if the inner function allow this kwarg
            kwargs["cache_dir"] = cache_dir
        # Get hash from the function parameters
        params = (func.__name__,) + args + tuple(kwargs.items())
        h = hashlib.sha256("".join(str(elem) for elem in params).encode()).hexdigest()
        cache_filename = f"gia-{h}"
        dirname = os.path.expanduser(cache_dir)
        os.makedirs(dirname, exist_ok=True)
        cache_path = os.path.join(dirname, cache_filename)

        if load_from_cache and os.path.exists(cache_path):
            logger.info(f"Loading cache ({cache_path})")
            return torch.load(cache_path)

        result = func(*args, **kwargs)

        # Save the result to cache
        torch.save(result, cache_path)

        return result

    return wrapper


def ufloat_encoder(obj):
    if isinstance(obj, Variable):
        return {"__ufloat__": True, "n": obj.n, "s": obj.s}
    return obj


def ufloat_decoder(dct):
    if "__ufloat__" in dct:
        return ufloat(dct["n"], dct["s"])
    return dct
