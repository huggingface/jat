from typing import Any, Dict, Iterable, List

import numpy as np
from torch.utils.data import Dataset


class DatasetDict(dict, Dataset):
    """
    A dataset that contains multiple fields.

    Example:
    >>> dataset = DatasetDict({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> dataset[0]
    {'a': 1, 'b': 4}
    >>> dataset[1]
    {'a': 2, 'b': 5}
    >>> dataset["b"]
    [4, 5, 6]
    >>> len(dataset)
    3
    >>> dataset.pop("a")
    [1, 2, 3]
    >>> dataset[0]
    {'b': 4}
    >>> dataset[:-1]
    {'b': [4, 5]}
    >>> len(dataset)
    3
    >>> dataset.keys()
    dict_keys(['b'])
    >>> dataset.values()
    dict_values([[4, 5, 6]])
    >>> dataset.items()
    dict_items([('b', [4, 5, 6])])
    >>> list(dataset)
    [{'b': 4}, {'b': 5}, {'b': 6}]
    """

    def __getitem__(self, key_or_index) -> Dict[str, Any]:
        if isinstance(key_or_index, (int, slice)):
            return {key: value[key_or_index] for key, value in self.items()}
        else:
            return super().__getitem__(key_or_index)

    def __len__(self) -> int:
        return len(next(iter(self.values())))

    def __iter__(self) -> Dict[str, Any]:
        for index in range(len(self)):
            yield self[index]


def is_text(x: Iterable) -> bool:
    """
    Check if input is text.

    It checks if the input a array of strings.
    """
    return all(isinstance(s, str) for s in x)


def is_image(x: np.ndarray) -> bool:
    """
    Check if input is an image.

    Returns True if the input has 4 dimensions.
    """
    return x.ndim == 4


def is_continuous(x: np.ndarray) -> bool:
    """
    Check if input is continous.

    Returns True if the dtype is float32 or float64.
    """
    return x.dtype in [np.float32, np.float64]


def is_discrete(x: np.ndarray) -> bool:
    """
    Check if input is discrete.

    Returns True if the dtype is int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
    """
    return x.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]


def stack_with_padding(x: List[np.ndarray], padding_value: int = 0, side: str = "left") -> np.ndarray:
    """
    Stack a list of numpy arrays with padding along the first dimension.

    Args:
        x (List[np.ndarray]): A list of numpy arrays to be stacked.
        padding_value (int, optional): The value to use for padding the arrays. Defaults to 0.
        side (str, optional): The side of the arrays to pad. Can be either "left" or "right". Defaults to "left".

    Returns:
        np.ndarray: The stacked array with padding.

    Raises:
        AssertionError: If the input arrays have different dimensions except for the first one.
    """
    # Check that all array have the same dimensions except the first one
    assert all(arr.shape[1:] == x[0].shape[1:] for arr in x)

    # Get the shape in the first dimension
    shape = (max(arr.shape[0] for arr in x), *x[0].shape[1:])

    # Initialize the stacked array with padding_value
    stacked = np.full((len(x), *shape), padding_value, dtype=x[0].dtype)

    # Fill the stacked array with input arrays
    for idx, arr in enumerate(x):
        if side == "left":
            stacked[idx, : arr.shape[0]] = arr
        elif side == "right":
            stacked[idx, -arr.shape[0] :] = arr
        else:
            raise ValueError("Invalid value for 'side' argument. Must be 'left' or 'right'.")

    return stacked
