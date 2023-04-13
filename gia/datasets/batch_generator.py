import random
from typing import Dict, List

import numpy as np
from datasets import get_dataset_config_names
from torch.utils.data import DataLoader

from gia.config import DatasetArguments
from gia.processor.multimodal_processor import MultimodalProcessor
from gia.utils.utils import cache_decorator

from .dataset_dict import DatasetDict
from .gia_dataset import load_gia_dataset
from .mixed_dataloader import MixedDataLoader


def stack_with_padding(x: List[np.ndarray], padding_value: int = 0, side: str = "left") -> np.ndarray:
    """
    Stack a list of numpy arrays with padding along the first dimension.

    Args:
        x (List[np.ndarray]): A list of numpy arrays to be stacked.
        padding_value (int, optional): The value to use for padding the arrays. Defaults to 0.
        side (str, optional): The side of the arrays to pad. Can be either "left" or "right". Defaults to "left".

    Returns:
        A tuple of two numpy arrays:
        - stacked: The stacked array with padding.
        - mask: A boolean mask indicating which values in the stacked array correspond to original input
            arrays (True) or padding (False).

    Raises:
        AssertionError: If the input arrays have different dimensions except for the first one.
    """
    # Check that all array have the same dimensions except the first one
    assert all(arr.shape[1:] == x[0].shape[1:] for arr in x)

    # Get the shape in the first dimension
    shape = (max(arr.shape[0] for arr in x), *x[0].shape[1:])

    # Initialize the stacked array with padding_value
    stacked = np.full((len(x), *shape), padding_value, dtype=x[0].dtype)
    mask = np.zeros((len(x), *shape), dtype=bool)

    # Fill the stacked array with input arrays
    for idx, arr in enumerate(x):
        if side == "left":
            stacked[idx, : arr.shape[0]] = arr
            mask[idx, : arr.shape[0]] = True
        elif side == "right":
            stacked[idx, -arr.shape[0] :] = arr
            mask[idx, -arr.shape[0] :] = True
        else:
            raise ValueError("Invalid value for 'side' argument. Must be 'left' or 'right'.")

    return stacked, mask


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
            dataset[f"{key}_loss_mask"] = np.zeros(shape, dtype=bool)
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
    dataset = {key: stack_with_padding(dataset[key]) for key in dataset.keys()}  # dict of tuples (data, mask)

    # Create a new entries for the masks
    data = {key: value[0] for key, value in dataset.items()}
    # We di about the attention mask for loss mask, since they are equal to the attention mask
    # of the modality they are associated with.
    attention_mask = {
        f"{key}_attention_mask": value[1] for key, value in dataset.items() if not key.endswith("loss_mask")
    }
    # Small hack here, for image patches, we need to turn the attention mask into the right shape
    if "image_observations_attention_mask" in attention_mask:
        attention_mask.pop("image_observations_attention_mask")
        mask = attention_mask.pop("patches_positions_attention_mask")
        # From shape (batch_size, seq_len, num_patches, 2, 2) to (batch_size, seq_len, num_patches)
        # Note that all the values are egals in the last two dimensions, so we can just take the first one
        mask = mask[:, :, :, 0, 0]
        attention_mask["image_observations_attention_mask"] = mask

    return {**data, **attention_mask}


@cache_decorator
def load_batched_dataset(task_name: str, args: DatasetArguments) -> DatasetDict:
    """
    Load a GIA dataset, tokenize, and generate batches.

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
                   'continuous_observations_loss_mask', 'continuous_actions_loss_mask', 'rewards_attention_mask',
                   'dones_attention_mask', 'continuous_observations_attention_mask',
                   'continuous_actions_attention_mask'])
        >>> dataset["continuous_observations"].shape
        (4074, 28, 27)
    """
    dataset = load_gia_dataset(task_name, args.load_from_cache)
    dataset = generate_batch(dataset, args)
    return DatasetDict(dataset)


@cache_decorator
def load_prompt_dataset(task_name: str, args: DatasetArguments) -> DatasetDict:
    """
    Generate a dataset of prompts.

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
    dataset = load_gia_dataset(task_name, args.load_from_cache)

    processor = MultimodalProcessor(args)
    # Preprocess the dataset (tokenize and extract patches)
    observation_keys = [key for key in dataset.keys() if key.endswith("observations")]
    action_keys = [key for key in dataset.keys() if key.endswith("actions")]  # should be only one
    dataset.update(processor({key: dataset[key] for key in observation_keys + action_keys}))

    episode_ends = np.where(dataset["dones"])[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    episode_indices = [np.arange(start, end + 1) for start, end in zip(episode_starts, episode_ends)]
    batch = {key: [value[episode] for episode in episode_indices] for key, value in dataset.items()}
    batch = {key: stack_with_padding(value, side="right") for key, value in batch.items()}

    # Create a new entries for the masks
    data = {key: value[0] for key, value in batch.items()}
    # We don't care about the attention mask for loss mask, since they are equal
    # to the attention mask of the modality they are associated with.
    attention_mask = {
        f"{key}_attention_mask": value[1] for key, value in batch.items() if not key.endswith("loss_mask")
    }
    # Small hack here, for image patches, we need to turn the attention mask into the right shape
    if "image_observations_attention_mask" in attention_mask:
        attention_mask.pop("image_observations_attention_mask")
        mask = attention_mask.pop("patches_positions_attention_mask")
        # From shape (batch_size, seq_len, num_patches, 2, 2) to (batch_size, seq_len, num_patches)
        # Note that all the values are egals in the last two dimensions, so we can just take the first one
        mask = mask[:, :, :, 0, 0]
        attention_mask["image_observations_attention_mask"] = mask

    return DatasetDict({**data, **attention_mask})


def get_dataloader(args: DatasetArguments) -> DataLoader:
    """
    Get the dataloader for the tokenized datasets.

    Args:
        args (DatasetArguments): The dataset arguments.

    Returns:
        DataLoader: The dataloader.

    Example:
        >>> from gia.datasets import get_dataloader
        >>> from gia.config import DatasetArguments
        >>> args = DatasetArguments(task_names=["babyai-go-to", "mujoco-ant"], shuffle=True, batch_size=2)
        >>> dataloader = get_dataloader(args)
        >>> iterator = iter(dataloader)
        >>> batch = next(iterator)
        >>> batch.keys()
        dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions',
            'continuous_observations_loss_mask', 'continuous_actions_loss_mask', 'rewards_attention_mask',
            'dones_attention_mask', 'continuous_observations_attention_mask', 'continuous_actions_attention_mask'])
        >>> batch["continuous_observations"].shape
        torch.Size([2, 28, 27])
        >>> batch = next(iterator)
        >>> batch.keys()
        dict_keys(['rewards', 'dones', 'text_observations', 'discrete_observations', 'image_observations',
            'discrete_actions', 'patches_positions', 'text_observations_loss_mask', 'discrete_observations_loss_mask',
            'image_observations_loss_mask', 'discrete_actions_loss_mask', 'rewards_attention_mask',
            'dones_attention_mask', 'text_observations_attention_mask', 'discrete_observations_attention_mask',
            'discrete_actions_attention_mask', 'image_observations_attention_mask'])
        >>> batch["image_observations"].shape
        torch.Size([2, 39, 16, 3, 16, 16])
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
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle) for dataset in datasets]
    return MixedDataLoader(loaders, shuffle=args.shuffle)
