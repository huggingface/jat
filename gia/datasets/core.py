import random
from typing import Dict, List, Union

import numpy as np
import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from torch import Tensor

from .utils import DatasetDict


def get_task_name_list(task_names: Union[str, List[str]]) -> List[str]:
    """
    Get the list of task names from a list of task names or prefixes.

    Args:
        task_names (Union[str, List[str]]): Name or list of names or prefixes of the tasks to load. See the
            available tasks in https://huggingface.co/datasets/gia-project/gia-dataset

    Raises:
        ValueError: If a task name is not found in the dataset.

    Returns:
        List[str]: List of task names.

    Example:
        >>> get_task_name_list(["mujoco", "atari-pong"])
        ['atari-pong', 'mujoco-doublependulum', 'mujoco-pendulum', 'mujoco-hopper', 'mujoco-ant',
         'mujoco-halfcheetah', 'mujoco-walker', 'mujoco-reacher', 'mujoco-swimmer']
    """
    # Convrt to list if needed
    task_names = [task_names] if isinstance(task_names, str) else task_names
    # Get all task names from gia dataset
    all_tasks = set(get_dataset_config_names("gia-project/gia-dataset"))
    # If the task name is a domain, load all the tasks of that domain
    for task_name in task_names:
        if task_name == "all":
            task_names.extend(all_tasks)
            task_names.remove("all")
        elif task_name not in all_tasks:
            # task_name is actaully a prefix
            prefix = task_name
            tasks = [task for task in all_tasks if task.startswith(prefix)]
            if len(tasks) > 0:
                task_names.extend(tasks)
                task_names.remove(task_name)
            else:
                raise ValueError(f"Task {task_name} not found in the dataset.")
    return task_names


def generate_prompts(dataset: Dataset, num_prompts: int, p_end: float = 0.1, max_prompt_len: int = 10) -> Dataset:
    """
    Generate prompts from the dataset.

    Args:
        dataset (Dataset): Dataset to generate prompts from.
        num_prompts (int): Number of prompts to generate.
        p_end (float, optional): Probability of generating a prompt from the end of the episode. Defaults to 0.1.
        max_prompt_len (int, optional): Maximum length of the prompt. Defaults to 10.
    """
    ep_lens = [len(ep[next(iter(ep))]) for ep in dataset]
    prompt_ep_idxs = random.choices(range(len(dataset)), k=num_prompts)
    from_ends = random.choices([True, False], k=num_prompts, weights=[p_end, 1 - p_end])
    prompt_lengths = random.choices(range(1, max_prompt_len + 1), k=num_prompts)  # will be clipped later if necessary
    starts = []
    for ep_idx, from_end, prompt_length in zip(prompt_ep_idxs, from_ends, prompt_lengths):
        max_start = max(0, ep_lens[ep_idx] - prompt_length)
        if from_end:
            starts.append(max_start)
        else:
            starts.append(random.randint(0, max_start))

    prompts_ep = dataset.select(prompt_ep_idxs)

    def dict_slice(x: Dict[str, List], idx: int) -> Dict[str, List]:
        start, length = starts[idx], prompt_lengths[idx]
        return {key: x[key][start : start + length] for key in x}

    prompt_dataset = prompts_ep.map(dict_slice, with_indices=True)
    return prompt_dataset


def needs_prompt(task_name: str) -> bool:
    """
    Check if the task needs prompt.
    """
    is_mujoco = task_name.startswith("mujoco")
    is_metaworld = task_name.startswith("metaworld")
    return is_mujoco or is_metaworld


def concatenate_datasets(datasets: List[Dataset]) -> DatasetDict:
    """
    Concatenate a list of datasets into a single dataset.

    Args:
        datasets (List[Dataset]): List of datasets to concatenate.

    Returns:
        DatasetDict: Concatenated dataset.

    Example:
        >>> dataset1 = Dataset.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> dataset2 = Dataset.from_dict({"a": [7, 8, 9], "c": [10, 11, 12]})
        >>> concatenate_datasets([dataset1, dataset2])
        {'a': [1, 2, 3, 7, 8, 9],
         'b': [4, 5, 6, None, None, None],
         'c': [None, None, None, 10, 11, 12]}
    """
    all_keys = set().union(*[d.column_names for d in datasets])  # Datasets can have different keys
    lengths = [len(d) for d in datasets]
    dataset = {key: [] for key in all_keys}
    for key in all_keys:
        for dataset_idx in range(len(datasets)):
            if key in datasets[dataset_idx].column_names:
                dataset[key].extend(datasets[dataset_idx][key])
            else:
                dataset[key].extend([None] * lengths[dataset_idx])
    return DatasetDict(dataset)


def load_gia_dataset(
    task_names: Union[str, List[str]],
    split: str = "all",
    p_prompt: float = 0.25,
    p_end: float = 0.5,
    max_prompt_len: int = 10,
    load_from_cache: bool = True,
) -> DatasetDict:
    """
    Load the GIA dataset.

    Args:
        task_names (str): Name or or list of names or prefix of the task(s) to load. See the available tasks in
            https://huggingface.co/datasets/gia-project/gia-dataset.
        split (str): Split of the dataset to load. One of "all", "train" or "test".
        p_prompt (float, optional): Probability of generating a prompt. Defaults to 0.25.
        p_end (float, optional): Probability that the prompt is generated from the end of the episode. Defaults to 0.5.
        max_prompt_len (int, optional): Maximum length of the prompt. Defaults to 10.
        load_from_cache (bool, optional): Whether to load the dataset from the local cache or to download it again.
            Defaults to True.

    Returns:
        DatasetDict: A dictionary containing the dataset. The keys are
            the type of the data (e.g. "continuous_observations", "discrete_actions", etc.)
            and the values are the data.

    Example:
        >>> dataset = load_gia_dataset("mujoco-ant", "train")

    """
    task_names = get_task_name_list(task_names)
    download_mode = "force_redownload" if not load_from_cache else None
    datasets = [
        load_dataset("gia-project/gia-dataset", task_name, split=split, download_mode=download_mode)
        for task_name in task_names
    ]
    # Print the first component of the first observation of every episode
    for dataset_idx in range(len(datasets)):
        dataset = datasets[dataset_idx]
        if needs_prompt(task_names[dataset_idx]):
            is_prompted = random.choices([True, False], weights=[p_prompt, 1 - p_prompt], k=len(dataset))
            prompts = generate_prompts(dataset, sum(is_prompted), p_end, max_prompt_len)  # [p0, p1, p2, ...]
            prompts_iter = iter(prompts)
            prompts = [next(prompts_iter) if p else None for p in is_prompted]  # [p0, None, None, p1, None, p2, ...]

            def cat_prompt_left(sample, idx):
                if prompts[idx] is not None:
                    for key in sample:
                        sample[key] = prompts[idx][key] + sample[key]
                return sample

            # Concatenate the prompt left of the episode
            datasets[dataset_idx] = dataset.map(cat_prompt_left, with_indices=True)

    # Concatenate the datasets
    dataset = concatenate_datasets(datasets)
    return dataset


def collate_fn(batch: List[Dict[str, List]]) -> Dict[str, Tensor]:
    """
    Collate function for the dataloader.

    Args:
        batch (List[Dict[str, List]]): List of samples.

    Returns:
        Dict[str, Tensor]: Collated batch.
    """
    # All samples must have the same keys
    keys = batch[0].keys()
    d = {}

    for key in keys:
        val = [sample[key] for sample in batch]
        if key == "patches":
            B, L = len(val), len(val[0])
            collated_data = torch.zeros((B, L, 4, 16, 16), dtype=torch.uint8)  # hard-coded for now
            for i, ep in enumerate(val):
                for j, patch in enumerate(ep):
                    patch = torch.as_tensor(patch)
                    shape = patch.shape
                    collated_data[i, j, : shape[0], : shape[1], : shape[2]] = patch
            val = collated_data

        if isinstance(val[0][0], np.ndarray):
            val = np.array(val)  # to avoid creating a tensor from a list of arrays
        d[key] = torch.as_tensor(val)

    return d
