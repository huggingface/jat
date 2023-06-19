import random
import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset


T = TypeVar("T", List, np.ndarray)


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
    if isinstance(task_names, str):
        if "," in task_names:
            task_names = task_names.split(",")
        else:
            task_names = [task_names]
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


def needs_prompt(task_name: str) -> bool:
    """
    Check if the task needs prompt. Usually, tasks that need prompt are RL ones.

    Args:
        task_name (str): Name of the task.

    Returns:
        bool: True if the task needs prompt, False otherwise.
    """
    is_atari = task_name.startswith("atari")
    is_babyai = task_name.startswith("babyai")
    is_conceptual_captions = task_name == "conceptual_captions"
    is_metaworld = task_name.startswith("metaworld")
    is_mujoco = task_name.startswith("mujoco")
    if is_atari or is_babyai or is_metaworld or is_mujoco:
        return True
    elif is_conceptual_captions:
        return False
    else:
        warnings.warn(f"Wether {task_name} needs prompt is unknown. Assuming it does not need prompt.")
        return False


class Prompter:
    def __init__(
        self,
        dataset: Dataset,
        p_prompt: float = 0.25,
        p_end: float = 0.1,
        min_prompt_len: int = 1,
        max_prompt_len: int = 1024,
    ) -> None:
        self.dataset = dataset
        self.ep_lens = [len(ep[next(iter(ep))]) for ep in self.dataset]
        self.p_prompt = p_prompt
        self.p_end = p_end
        self.min_prompt_len = min_prompt_len
        self.max_prompt_len = max_prompt_len

    def generate_prompts(self, num_prompts: int) -> Dict[str, List]:
        prompt_ep_idxs = random.choices(range(len(self.dataset)), k=num_prompts)
        from_ends = random.choices([True, False], k=num_prompts, weights=[self.p_end, 1 - self.p_end])
        ep_lens = [self.ep_lens[idx] for idx in prompt_ep_idxs]
        prompt_lengths = random.choices(range(self.min_prompt_len, self.max_prompt_len + 1), k=num_prompts)
        prompts = self.dataset.select(prompt_ep_idxs).to_dict()
        for idx in range(num_prompts):
            max_start = max(0, ep_lens[idx] - prompt_lengths[idx])
            start = max_start if from_ends[idx] else random.randint(0, max_start)
            for key in prompts:
                prompts[key][idx] = prompts[key][idx][start : start + prompt_lengths[idx]]
        return prompts

    @staticmethod
    def _cat(x: T, y: T) -> T:
        if isinstance(x, list) and isinstance(y, list):
            return x + y
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.concatenate((x, y))
        else:
            raise ValueError("x and y must be either lists or numpy arrays")

    def prompt(self, examples: Dict[str, List]) -> Dict[str, List]:
        num_examples = len(examples[next(iter(examples))])
        to_prompt_idxs = [idx for idx in range(num_examples) if random.random() < self.p_prompt]
        prompts = self.generate_prompts(len(to_prompt_idxs))
        for idx, to_prompt_idx in enumerate(to_prompt_idxs):
            for key in examples:
                examples[key][to_prompt_idx] = self._cat(prompts[key][idx], examples[key][to_prompt_idx])
        return examples


def load_and_process_dataset(
    task_names: List[str],
    split: str,
    processor: Callable,
    load_from_cache_file: bool = True,
    preprocessing_num_workers: Optional[int] = None,
):
    """
    Load, prompt and process the dataset.

    Args:
        task_names (List[str]): List of task names to load.
        split (str): Split of the dataset to load.
        processor (Callable): Processor to process the dataset.
        load_from_cache_file (bool, optional): Whether to load the dataset from cache or not. Defaults to True.
        preprocessing_num_workers (Optional[int], optional): Number of workers for preprocessing. Defaults to None.

    Returns:
        Dataset: Processed dataset.
    """

    train_datasets = {
        task_name: load_dataset("gia-project/gia-dataset", task_name, split=split) for task_name in task_names
    }
    prompters = {
        task_name: Prompter(dataset) for task_name, dataset in train_datasets.items() if needs_prompt(task_name)
    }

    def prompt_and_process(example, prompter):
        if prompter is not None:
            return processor(**prompter.prompt(example))
        else:
            return processor(**example)

    train_datasets = {
        task_name: dataset.map(
            partial(prompt_and_process, prompter=prompters.get(task_name)),
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=20,  # lower this from 1000 to 20 avoid OOM
            num_proc=preprocessing_num_workers,
            load_from_cache_file=load_from_cache_file,
        )
        for task_name, dataset in train_datasets.items()
    }

    train_dataset = concatenate_datasets(list(train_datasets.values()))

    return train_dataset
