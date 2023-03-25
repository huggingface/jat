import hashlib
import os
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset

from .batch_generator import BatchGenerator
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


def load_gia_dataset(
    task_name: str,
    seq_len: int = 1024,
    p_prompt: float = 0.25,
    p_end: float = 0.5,
    patch_size: int = 16,
    mu: float = 100,
    M: float = 256,
    nb_bins: int = 1024,
    token_shift: int = 32_000,
    use_sepatator: bool = True,
    load_from_cache_file: bool = True,
) -> DatasetDict:
    """
    Load a GIA dataset.

    Args:
        task_name (str): Name of the task to load. See the available tasks
            in https://huggingface.co/datasets/gia-project/gia-dataset
        seq_len (int, optional): The length of the sequence of embeddings to be generated. Defaults to 1024.
        p_prompt (float, optional): Probability of adding a prompt to a sequence. Defaults to 0.25.
        p_end (float, optional): Probability that the prompt is the end of the episode. Defaults to 0.5.
        patch_size (int, optional): The size of the square patch to be extracted from the image. Defaults to 16.
        mu (float, optional): μ parameter for the μ-law companding. Defaults to 100.
        nb_bins (int, optional): Number of bins for the discretization of continuous values. Defaults to 1024.
        token_shift (int, optional): Shift for the discrete tokens. Defaults to 32_000.
        load_from_cache_file (bool, optional): Whether to load the dataset from the cache. Defaults to True.
        use_sepatator (bool, optional): Whether to use a separator token between the observations and the actions.
            Defaults to True.

    Example:
        >>> dataset = load_gia_dataset("babyai-go-to")
        >>> dataset[0]["observations/image"].shape
        torch.Size([56, 3, 56, 56])
    """
    # Get hash from the function parameters
    params = (task_name, p_prompt, p_end, seq_len, patch_size, mu, M, nb_bins, token_shift, use_sepatator)
    h = hashlib.sha256("".join(str(elem) for elem in params).encode()).hexdigest()
    cache_filename = f"gia-{h}"
    dirname = os.path.expanduser("~/.cache/huggingface/datasets")
    os.makedirs(dirname, exist_ok=True)
    cache_path = os.path.join(dirname, cache_filename)

    if load_from_cache_file and os.path.exists(cache_path):
        print(f"Loading cached dataset ({cache_path})")
        return torch.load(cache_path)

    dataset = load_dataset("gia-project/gia-dataset", task_name, split="train")
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

    # Pack the the dataset into batches of sequences
    batch_generator = BatchGenerator(seq_len, p_prompt, p_end, patch_size, mu, M, nb_bins, token_shift, use_sepatator)
    dataset = batch_generator(dataset)
    dataset = DatasetDict(dataset)  # convert to a DatasetDict
    torch.save(dataset, cache_path)  # save into cache file
    return dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = load_gia_dataset("babyai-go-to", load_from_cache_file=False)
    daltaloader = DataLoader(dataset, batch_size=64)
    for batch in tqdm(daltaloader):
        for key, value in batch.items():
            tqdm.write(f"{key}: {value.shape} {value.dtype}")
        tqdm.write("---")
