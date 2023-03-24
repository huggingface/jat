import hashlib
import os
import random
from typing import Dict, Iterable, List

import numpy as np
import torch
from datasets import load_dataset

from gia.datasets.dataset_dict import DatasetDict
from gia.processor.multimodal_processor import MultimodalProcessor


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


class BatchGenerator:
    """
    Batch generator for the GIA dataset.

    Args:
        seq_len (int, optional): The length of the sequence of embeddings to be generated. Defaults to 1024.
        p_prompt (float, optional): Probability of adding a prompt to a sequence. Defaults to 0.25.
        p_end (int, optional): The probability that the prompt is taken from the end of an episode. Defaults to 0.5.
        patch_size (int, optional): The size of the square patch to be extracted from the image. Defaults to 16.
        mu (float, optional): μ parameter for the μ-law companding. Defaults to 100.
        M (float, optional): M parameter for the μ-law companding. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization of continuous values. Defaults to 1024.
        token_shift (int, optional): Shift for the discrete tokens. Defaults to 32_000.
        use_separator (bool, optional): Whether to use a separator token between the observations and actions.
            Defaults to True.

    Input:
        - dataset (Dict[str, np.ndarray]): A dictionary containing the dataset.
            The dictionary must contain the following keys:
                - "observations[*]": Observations (possibly many keys starting with "observations")
                - "actions": Actions
                - "dones": Dones flags

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the processed dataset with keys for observations,
            actions, and masks.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        p_prompt: float = 0.25,
        p_end: float = 0.5,
        patch_size: int = 16,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        token_shift: int = 32_000,
        use_separator: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.p_prompt = p_prompt
        self.p_end = p_end
        self.patch_size = patch_size
        self.use_separator = use_separator
        self.processor = MultimodalProcessor(mu, M, nb_bins, patch_size, token_shift)

    @staticmethod
    def stack_with_padding(x: List[np.ndarray], padding_value: int = 0) -> np.ndarray:
        """
        Stack a list of numpy arrays along a new axis with padding, aligning each input array to the top-left corner.

        Args:
            x (List[np.ndarray]): A list of numpy arrays to stack. The arrays may have different shapes.
            padding_value (int, optional): The value to use for padding the stacked array. Defaults to 0.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - stacked (np.ndarray): A stacked numpy array with padding, such that the original arrays are aligned
                    to the top-left corner.
                - mask (np.ndarray): A boolean numpy array of the same shape as the stacked array, with True
                    indicating the positions of the original data and False indicating the padding.
        """

        # Find the max shape of arrays in x
        max_shape = np.array([arr.shape for arr in x]).max(axis=0)

        # Initialize the stacked array with padding_value
        stacked = np.full((len(x), *max_shape), padding_value, dtype=x[0].dtype)
        mask = np.zeros((len(x), *max_shape), dtype=bool)

        # Fill the stacked array with input arrays
        for idx, arr in enumerate(x):
            slices = tuple(slice(0, dim) for dim in arr.shape)
            stacked[idx][slices] = arr
            mask[idx][slices] = True

        return stacked, mask

    @staticmethod
    def get_num_patches(x: np.ndarray, patch_size: int) -> int:
        """
        Calculate the total number of non-overlapping patches in a 2D input array or image.

        Args:
            x (np.ndarray): A 2D numpy array or image with shape (batch_size, channels, height, width).
            patch_size (int): The size of the square patch to be extracted.

        Returns:
            int: The total number of non-overlapping patches in the input array or image.
        """
        _, _, height, width = x.shape
        return (height // patch_size) * (width // patch_size)

    def __call__(self, dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Preprocess the dataset (tokenize and extract patches)
        observation_keys = [key for key in dataset.keys() if key.endswith("observations")]
        action_keys = [key for key in dataset.keys() if key.endswith("actions")]  # should be only one
        dataset.update(self.processor({key: dataset[key] for key in observation_keys + action_keys}))

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
        if self.use_separator:
            num_emb_per_interaction += 1

        num_interractions_per_seq = self.seq_len // num_emb_per_interaction

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
            if random.random() < self.p_prompt:
                num_prompt_interactions = random.randint(1, num_interractions_per_seq)
                num_interractions = num_interractions_per_seq - num_prompt_interactions
                prompt_ep_idx = random.randint(0, len(ep_end_idxs) - 1)
                if random.random() < self.p_end:  # Prompt at the end of the episode
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
        dataset = {key: self.stack_with_padding(dataset[key]) for key in dataset.keys()}  # dict of tuples (data, mask)

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
