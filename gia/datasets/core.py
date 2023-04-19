import random
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from torch.utils.data import ConcatDataset, Dataset

from gia.config import DatasetArguments
from gia.processor.multimodal_processor import MultimodalProcessor
from gia.utils.utils import cache_decorator

from .utils import (
    DatasetDict,
    is_continuous,
    is_discrete,
    is_image,
    is_text,
    stack_with_padding,
)


def load_task_dataset(task_name: str, load_from_cache: bool = True) -> DatasetDict:
    """
    Load the dataset for a single task.

    Args:
        task_name (str): Name of the task to load. See the available tasks
            in https://huggingface.co/datasets/gia-project/gia-dataset

    Returns:
        DatasetDict: A dictionary containing the dataset. The keys are
            the type of the data (e.g. "continuous_observations", "discrete_actions", etc.)
            and the values are the data.

    Example:
        >>> dataset = load_task_dataset("mujoco-ant")
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


def generate_batch(dataset: Dict[str, np.ndarray], args: DatasetArguments) -> Dict[str, np.ndarray]:
    """
    Generates a batch of sequences from a multimodal dataset by preprocessing and concatenating interactions.
    Optionally includes prompts at the beginning of sequences with a specified probability.

    Args:
        dataset (Dict[str, np.ndarray]): A dictionary containing the dataset with keys for observations and actions.
        args (DatasetArguments): The dataset arguments.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the preprocessed dataset with keys for observations and actions.
    """
    processor = MultimodalProcessor(args)
    # Preprocess the dataset (tokenize and extract patches)
    observation_keys = [key for key in dataset.keys() if key.endswith("observations")]
    action_keys = [key for key in dataset.keys() if key.endswith("actions")]  # should be only one
    dataset.update(processor({key: dataset[key] for key in observation_keys + action_keys}))

    # Add loss mask: m = True if the token at index l is either from text or
    # from the logged action of an agent, and 0 otherwise
    for key in observation_keys + action_keys:
        shape = dataset[key].shape[:2]  # (batch_size, seq_len)
        if key.startswith("text") or key.endswith("actions"):
            dataset[f"{key}_loss_mask"] = np.ones(shape, dtype=bool)
        else:
            dataset[f"{key}_loss_mask"] = dataset[f"{key}_attention_mask"]
    # First, we need to compute the number of embeddings needed for the observations and actions.
    # At this point, there are 2 possibilities for the observations:
    # 1. The observation is anything but an image, then the value is tokenized
    # 2. The observation is an image, then the value is a tuple containing the patches and
    #    the corresponding positions
    # In both cases, the values are numpy arrays of shape (batch_size, seq_len, ...)
    # We compute the total number of embeddings for the observations and actions
    num_emb_per_interaction = sum(dataset[key].shape[1] for key in observation_keys + action_keys)

    # Add one embedding for the separator token
    if args.use_separator:
        num_emb_per_interaction += 1

    # Check that the sequence lenght is high enough to contain at least one interaction
    if args.seq_len < num_emb_per_interaction:
        raise ValueError(
            f"The sequence length ({args.seq_len}) is too short to contain at least one interaction "
            f"Use a sequence length of at least {num_emb_per_interaction}."
        )

    num_interractions_per_seq = args.seq_len // num_emb_per_interaction

    # We require at least two interactions per sequence to be able to prompt. TODO: why?
    # assert num_emb_per_interaction * 2 + 1 <= seq_len

    # From the done flags, compute the indices of the start of each episode
    ep_end_idxs = [i for i, done in enumerate(dataset["dones"]) if done]

    # We create sequences of interactions one by one.
    # In p_prompt % of the cases, we add a prompt at the beginning of the sequence.
    # We iterate until we don't have enough interactions left to fill a sequence
    current_ep_start = 0
    all_batches = []
    while True:
        if random.random() < args.p_prompt:
            num_prompt_interactions = random.randint(1, num_interractions_per_seq)
            num_interractions = num_interractions_per_seq - num_prompt_interactions
            prompt_ep_idx = random.randint(0, len(ep_end_idxs) - 1)
            if random.random() < args.p_end:  # Prompt at the end of the episode
                # Ensure that you don't take from the previous episode
                prev_ep_end_idx = ep_end_idxs[prompt_ep_idx - 1] if prompt_ep_idx > 0 else -1
                ep_end_idx = ep_end_idxs[prompt_ep_idx]
                prompt_ep_length = ep_end_idx - prev_ep_end_idx
                num_prompt_interactions = min(num_prompt_interactions, prompt_ep_length)
                prompt_indices = np.arange(ep_end_idx - num_prompt_interactions + 1, ep_end_idx + 1)
            else:  # Prompt anywhere in the episode
                # Get the range for the possible start of the prompt
                prev_ep_end_idx = ep_end_idxs[prompt_ep_idx - 1] if prompt_ep_idx > 0 else -1
                ep_end_idx = ep_end_idxs[prompt_ep_idx]
                prompt_ep_length = ep_end_idx - prev_ep_end_idx
                num_prompt_interactions = min(num_prompt_interactions, prompt_ep_length)
                prompt_start_idx = random.randint(prev_ep_end_idx + 1, ep_end_idx - num_prompt_interactions + 1)
                prompt_indices = np.arange(prompt_start_idx, prompt_start_idx + num_prompt_interactions)
        else:
            num_prompt_interactions = 0
            num_interractions = num_interractions_per_seq
            prompt_indices = np.arange(0)

        if current_ep_start + num_interractions > len(dataset["dones"]):  # No more interactions left
            break

        ep_indices = np.arange(current_ep_start, current_ep_start + num_interractions)

        indices = np.concatenate((prompt_indices, ep_indices))
        batch = {key: dataset[key][indices] for key in dataset.keys()}
        all_batches.append(batch)
        current_ep_start += num_interractions

    # Turn the list of dict into a dict of list
    dataset = {key: [batch[key] for batch in all_batches] for key in dataset.keys()}
    # Stack the list of arrays into a single array of shape (batch_size, seq_len, ...).
    # Use padding to fill when the sequence is shorter than others (happens when prompt is too short).
    dataset = {key: stack_with_padding(dataset[key]) for key in dataset.keys()}
    return dataset


@cache_decorator
def load_batched_dataset(task_name: str, args: DatasetArguments) -> DatasetDict:
    """
    Load the dataset for a single task and generate batches.

    Args:
        task_name (str): Name of the task to load. See the available tasks
            in https://huggingface.co/datasets/gia-project/gia-dataset
        args (DatasetArguments): The dataset arguments.

    Returns:
        DatasetDict: The dataset.

    Example:
        >>> from gia.datasets import load_batched_dataset
        >>> from gia.config import DatasetArguments
        >>> args = DatasetArguments()
        >>> dataset = load_batched_dataset("mujoco-ant", args)
        >>> len(dataset)
        4074
        >>> dataset.keys()
        dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions',
                   'continuous_observations_loss_mask', 'continuous_actions_loss_mask',
                   'continuous_observations_attention_mask', 'continuous_actions_attention_mask'])
        >>> dataset["continuous_observations"].shape
        (4074, 28, 27)
    """
    dataset = load_task_dataset(task_name, args.load_from_cache)
    dataset = generate_batch(dataset, args)
    return DatasetDict(dataset)


@cache_decorator
def load_prompt_dataset(task_name: str, args: DatasetArguments) -> DatasetDict:
    """
    Generate a dataset of prompts for a single task.

    Args:
        task_name (str): Name of the task to load. See the available tasks
            in https://huggingface.co/datasets/gia-project/gia-dataset
        args (DatasetArguments): The dataset arguments.

    Returns:
        DatasetDict: The dataset.

    Example:

        >>> from gia.datasets.batch_generator import load_prompt_dataset
        >>> from gia.config import DatasetArguments
        >>> args = DatasetArguments()
        >>> dataset = load_prompt_dataset("mujoco-ant", args)
        >>> dataset["continuous_observations"].shape
        (104, 1000, 27)
    """
    # Load the dataset
    dataset = load_task_dataset(task_name, args.load_from_cache)

    processor = MultimodalProcessor(args)
    # Preprocess the dataset (tokenize and extract patches)
    observation_keys = [key for key in dataset.keys() if key.endswith("observations")]
    action_keys = [key for key in dataset.keys() if key.endswith("actions")]  # should be only one
    dataset.update(processor({key: dataset[key] for key in observation_keys + action_keys}))

    episode_ends = np.where(dataset["dones"])[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    episode_indices = [np.arange(start, end + 1) for start, end in zip(episode_starts, episode_ends)]
    dataset = {key: [value[episode] for episode in episode_indices] for key, value in dataset.items()}
    dataset = {key: stack_with_padding(value, side="right") for key, value in dataset.items()}
    return DatasetDict(dataset)


def load_mixed_dataset(args: DatasetArguments) -> Dataset:
    """
    Load a dataset with multiple tasks.

    Args:
        args (DatasetArguments): The dataset arguments.

    Returns:
        Dataset: The dataset.

    Example:
        >>> from gia.datasets import load_mixed_dataset
        >>> from gia.config import DatasetArguments
        >>> args = DatasetArguments(task_names=["mujoco-ant", "babyai-go-to"])
        >>> dataset = load_mixed_dataset(args)
        >>> len(dataset)
        6981
        >>> dataset[0].keys()
        dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions',
            'continuous_observations_attention_mask', 'continuous_actions_attention_mask',
            'continuous_observations_loss_mask', 'continuous_actions_loss_mask'])
        >>> dataset[0]["continuous_observations"].shape
        (28, 27)
        >>> dataset[5000].keys()
        dict_keys(['rewards', 'dones', 'text_observations', 'discrete_observations', 'image_observations',
            'discrete_actions', 'text_observations_attention_mask', 'discrete_observations_attention_mask',
            'patches_positions', 'image_observations_attention_mask', 'discrete_actions_attention_mask',
            'text_observations_loss_mask', 'discrete_observations_loss_mask', 'image_observations_loss_mask',
            'discrete_actions_loss_mask'])
        >>> dataset[5000]["continuous_observations"]["image_observations"].shape
        (39, 16, 3, 16, 16)
    """
    task_names = [args.task_names] if isinstance(args.task_names, str) else args.task_names
    all_tasks = set(get_dataset_config_names("gia-project/gia-dataset"))  # get all task names from gia dataset
    all_domains = set(task_name.split("-")[0] for task_name in all_tasks)
    # If the task name is a domain, load all the tasks of that domain
    for task_name in task_names:
        if task_name in all_domains:
            task_names.extend([t for t in all_tasks if t.startswith(task_name)])
            task_names.remove(task_name)
        elif task_name == "all":
            task_names.extend(all_tasks)
            task_names.remove("all")
        elif task_name not in all_tasks:
            raise ValueError(f"Task {task_name} not found in the dataset.")

    datasets = [load_batched_dataset(task_name, args) for task_name in task_names]
    return ConcatDataset(datasets)


def collate_fn(batch: List[Dict[str, Any]]) -> Sequence[Dict[str, torch.Tensor]]:
    """
    Collate function for the dataloader. It converts the batch to list of a dictionaries of tensors.

    Args:
        batch (List[Dict[str, Any]]): The batch to collate.

    Returns:
        Sequence[Dict[str, torch.Tensor]]: The collated batch.
    """
    batch = [{key: torch.tensor(value) for key, value in d.items()} for d in batch]
    return batch
