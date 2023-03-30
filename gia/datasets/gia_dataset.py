from typing import Iterable

import numpy as np
from datasets import load_dataset

from .dataset_dict import DatasetDict


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


def load_gia_dataset(task_name: str, load_from_cache: bool = True) -> DatasetDict:
    """
    Load a GIA dataset.

    Args:
        task_name (str): Name of the task to load. See the available tasks
            in https://huggingface.co/datasets/gia-project/gia-dataset

    Returns:
        DatasetDict: A dictionary containing the dataset. The keys are
            the type of the data (e.g. "continuous_observations", "discrete_actions", etc.)
            and the values are the data.

    Example:
        >>> dataset = load_gia_dataset("mujoco-ant")
        >>> dataset.keys()
        dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions'])
        >>> dataset["continuous_observations"].shape
        (100000, 27)
    """
    download_mode = "force_redownload" if not load_from_cache else None
    dataset = load_dataset("gia-project/gia-dataset", task_name, split="train", download_mode=download_mode)
    # Convert the dataset to numpy arrays
    dataset = dataset.with_format("numpy")[:]

    # TODO: remove this when a better solution is found
    if "babyai" in task_name:
        # remove the "image" column as it is not used
        dataset.pop("images")

    # Rename keys to get the format "[type]_[observations or actions]"
    keys = list(dataset.keys())  # Avoid "Keys changed during iteration"
    for key in keys:
        if key not in ["actions", "dones", "rewards"]:
            observations = dataset.pop(key)
            if is_image(observations):
                observations = observations.astype(np.uint8)
                if np.argmin(observations.shape[1:]) == 2:  # channels last, make it channels first
                    observations = np.transpose(observations, (0, 3, 1, 2))
                assert np.argmin(observations.shape[1:]) == 0, "Channels error"
                dataset["image_observations"] = observations
            elif is_text(observations):
                dataset["text_observations"] = observations
            elif is_discrete(observations):
                dataset["discrete_observations"] = observations
            elif is_continuous(observations):
                dataset["continuous_observations"] = observations
            else:
                raise ValueError(f"Unknown observation type for {key}")
    # Do the same for actions.
    actions = dataset.pop("actions")
    if is_text(actions):
        dataset["text_actions"] = actions
    elif is_discrete(actions):
        dataset["discrete_actions"] = actions
    elif is_continuous(actions):
        dataset["continuous_actions"] = actions
    else:
        raise ValueError("Unknown action type.")

    return DatasetDict(dataset)  # convert to a DatasetDict
