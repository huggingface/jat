import random
import warnings
from typing import Dict, List, Union

from datasets import Dataset, get_dataset_config_names


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


def generate_prompts(
    dataset: Dataset, num_prompts: int, p_end: float = 0.1, min_prompt_len: int = 1, max_prompt_len: int = 1024
) -> Dataset:
    """
    Generate prompts from the dataset.

    Args:
        dataset (Dataset): Dataset to generate prompts from.
        num_prompts (int): Number of prompts to generate.
        p_end (float, optional): Probability of generating a prompt from the end of the episode. Defaults to 0.1.
        min_prompt_len (int, optional): Sets the minimum prompt length, defaulting to 1. Beware, if the sampled
            prompt is too short, the output prompt might also fall short of min_prompt_len.
        max_prompt_len (int, optional): Maximum length of the prompt. Defaults to 10.
    """
    ep_lens = [len(ep[next(iter(ep))]) for ep in dataset]
    prompt_ep_idxs = random.choices(range(len(dataset)), k=num_prompts)
    from_ends = random.choices([True, False], k=num_prompts, weights=[p_end, 1 - p_end])
    prompt_lengths = random.choices(
        range(min_prompt_len, max_prompt_len + 1), k=num_prompts
    )  # will be clipped later if necessary
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


def prompt_dataset(
    dataset: Dataset, p_prompt: float = 0.25, p_end: float = 0.1, min_prompt_len: int = 1, max_prompt_len: int = 1024
) -> Dataset:
    """
    Prompt the dataset.

    Args:
        dataset (Dataset): Dataset to prompt.
        p_prompt (float, optional): Probability of prompting an episode. Defaults to 0.25.
        p_end (float, optional): Probability of prompting from the end of the episode. Defaults to 0.1.
        min_prompt_len (int, optional): Minimum length of the prompt. Defaults to 1.
        max_prompt_len (int, optional): Maximum length of the prompt. Defaults to 1024.

    Returns:
        Dataset: Prompted dataset.
    """

    is_prompted = random.choices([True, False], weights=[p_prompt, 1 - p_prompt], k=len(dataset))
    prompts = generate_prompts(dataset, sum(is_prompted), p_end, min_prompt_len, max_prompt_len)  # [p0, p1, p2, ...]
    prompts_iter = iter(prompts)
    prompts = [next(prompts_iter) if p else None for p in is_prompted]  # [p0, None, None, p1, None, p2, ...]

    def cat_prompt_left(sample, idx):
        if prompts[idx] is not None:
            for key in sample:
                sample[key] = prompts[idx][key] + sample[key]
        return sample

    return dataset.map(cat_prompt_left, with_indices=True)  # concatenate the prompt left of the episode


def maybe_prompt_dataset(
    dataset: Dataset, p_prompt: float = 0.25, p_end: float = 0.1, min_prompt_len: int = 1, max_prompt_len: int = 1024
) -> Dataset:
    """
    First, check if dataset needs to be prompted, then, prompt it if needed.

    Args:
        dataset (Dataset): Dataset to prompt.
        p_prompt (float): Probability of prompting an episode.
        p_end (float): Probability of prompting from the end of the episode.
        min_prompt_len (int): Minimum length of the prompt.
        max_prompt_len (int): Maximum length of the prompt.

    Returns:
        Dataset: Prompted dataset if needed, the original dataset otherwise.
    """
    if needs_prompt(dataset.config_name):
        return prompt_dataset(dataset, p_prompt, p_end, min_prompt_len, max_prompt_len)
    else:
        return dataset


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

    def prompt(self, examples: Dict[str, List]) -> Dict[str, List]:
        num_examples = len(examples[next(iter(examples))])
        to_prompt_idxs = [idx for idx in range(num_examples) if random.random() < self.p_prompt]
        prompts = self.generate_prompts(len(to_prompt_idxs))
        for idx, to_prompt_idx in enumerate(to_prompt_idxs):
            for key in examples:
                examples[key][to_prompt_idx] = prompts[key][idx] + examples[key][to_prompt_idx]
        return examples
