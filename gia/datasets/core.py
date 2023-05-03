from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from datasets import get_dataset_config_names, load_dataset

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
    all_tasks = set(
        get_dataset_config_names(
            "gia-project/gia-dataset",
            revision="episode_structure",  # TODO: change this to the new dataset name
        )
    )
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


def load_gia_dataset(
    task_names: Union[str, List[str]], split: str = "all", load_from_cache: bool = True
) -> DatasetDict:
    """
    Load the GIA dataset.

    Args:
        task_names (str): Name or or list of names or prefix of the task(s) to load. See the available tasks in
            https://huggingface.co/datasets/gia-project/gia-dataset.
        split (str): Split of the dataset to load. One of "all", "train" or "test".
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
        load_dataset(
            "gia-project/gia-dataset",
            task_name,
            split=split,
            revision="episode_structure",  # TODO: change to "main" when the new version is released
            download_mode=download_mode,
        )
        for task_name in task_names
    ]
    # Convert the dataset to numpy arrays
    # datasets = [DatasetDict(dataset.with_format("numpy")[:]) for dataset in datasets]  # convert to a DatasetDict
    all_keys = set().union(*[d.column_names for d in datasets])
    datasets = {key: sum([d[key] if key in d.column_names else None for d in datasets], []) for key in all_keys}
    return DatasetDict(datasets)
