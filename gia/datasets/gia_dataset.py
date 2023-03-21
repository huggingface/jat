import random
from typing import Dict, List

import numpy as np
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader

from gia.processor.multimodal_processor import MultimodalProcessor


class BatchGenerator:
    """
    Batch generator for the GIA dataset.

    Args:
        seq_len (int): The length of the sequence to be generated.
        p_prompt (int): The probability of generating a prompt token.
        p_end (int): The probability the the prompt token is taken from the end of an episode.
        patch_size (int): The size of the square patch to be extracted from the image.
        mu (float): The mu parameter for the mu-law transformation.
        M (float): Mu-law companding parameter. Defaults to 256.
        nb_bins (int): The number of bins for the discretization of the continuous data.
        token_shift (int): The shift for the non-textual tokens.

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
        p_prompt: int = 0.25,
        p_end: int = 0.5,
        patch_size: int = 16,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        token_shift: int = 32_000,
    ) -> None:
        self.seq_len = seq_len
        self.p_prompt = p_prompt
        self.p_end = p_end
        self.patch_size = patch_size
        self.processor = MultimodalProcessor(mu, M, nb_bins, token_shift)

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
        stacked = np.full((len(x), *max_shape), padding_value)
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
        # Preprocess the dataset
        observation_keys = [key for key in dataset.keys() if key.startswith("observations")]
        dataset.update(self.processor({key: dataset[key] for key in observation_keys + ["actions"]}))

        # First, we need to compute the number of embeddings needed for the observations and actions.
        # At this point, there are 2 possibilities for the observations:
        # 1. The observation is anything but an image, then the input is tokenized
        # 2. The observation is an image
        # In both cases, the observations are stored in a numpy array of shape (batch_size, seq_len)
        # We need to first get the keys for both cases
        observation_keys = [key for key in dataset.keys() if key.startswith("observations")]
        tokenized_observation_keys = [key for key in observation_keys if dataset[key].ndim == 2]
        image_observation_keys = [key for key in observation_keys if dataset[key].ndim == 4]
        assert len(tokenized_observation_keys) + len(image_observation_keys) == len(observation_keys)

        # We first compute the number of embeddings for the tokenized observations
        # Get the number of embeddings for the observations
        num_obs_tokens = sum(dataset[key].shape[1] for key in tokenized_observation_keys)

        # Now, we need to compute the number of embeddings for the images.
        # The number of embeddings is the number of patches in the image.
        # In some rare cases, there are plus than one image in the observation, so we need to sum them.
        num_image_patches = sum(self.get_num_patches(dataset[key], self.patch_size) for key in image_observation_keys)

        # Finally, we can compute the total number of embeddings for the observations
        num_emb_per_observation = num_obs_tokens + num_image_patches

        # Now, we need to compute the number of embeddings for the actions.
        # It's simpler since there is only one key, and the action is not an image.
        num_emb_per_action = dataset["actions"].shape[1]

        # Compute the number of interactions per sequence
        num_emb_per_interaction = num_emb_per_observation + num_emb_per_action
        num_interractions_per_seq = self.seq_len // num_emb_per_interaction

        # We require at least two interactions per sequence to be able to prompt. TODO: why?
        # assert num_emb_per_interaction * 2 + 1 <= seq_len

        # From the done flags, compute the indexes of the start of each episode
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
                    prompt_indexes = np.arange(ep_end_idx - num_prompt_interactions + 1, ep_end_idx + 1)
                else:  # Prompt anywhere in the episode
                    # Get the range for the possible start of the prompt
                    prev_ep_end_idx = ep_end_idxs[prompt_ep_idx - 1] if prompt_ep_idx > 0 else -1
                    ep_end_idx = ep_end_idxs[prompt_ep_idx]
                    prompt_ep_length = ep_end_idx - prev_ep_end_idx
                    num_prompt_interactions = min(num_prompt_interactions, prompt_ep_length)
                    prompt_start_idx = random.randint(prev_ep_end_idx + 1, ep_end_idx - num_prompt_interactions + 1)
                    prompt_indexes = np.arange(prompt_start_idx, prompt_start_idx + num_prompt_interactions)
            else:
                num_prompt_interactions = 0
                num_interractions = num_interractions_per_seq
                prompt_indexes = np.arange(0)

            if current_ep_start + num_interractions > len(dataset["dones"]):  # No more interactions left
                break

            ep_indexes = np.arange(current_ep_start, current_ep_start + num_interractions)

            indexes = np.concatenate((prompt_indexes, ep_indexes))
            batch = {key: dataset[key][indexes] for key in observation_keys + ["actions"]}
            all_batches.append(batch)
            current_ep_start += num_interractions

        # Turn the list of dict into a dict of list
        dataset = {key: [batch[key] for batch in all_batches] for key in observation_keys + ["actions"]}
        # Stack the list of arrays into a single array of shape (batch_size, seq_len, ...).
        # Use padding to fill when the sequence is shorter than others.
        dataset = {key: self.stack_with_padding(dataset[key]) for key in dataset.keys()}  # dict of tuples (data, mask)

        # Create a new entries for the masks
        data = {key: value[0] for key, value in dataset.items()}
        masks = {f"{key}_mask": value[1] for key, value in dataset.items()}
        return {**data, **masks}


def load_gia_dataset(
    task_name: str,
    p_prompt: float = 0.25,
    p_end: float = 0.5,
    seq_len: int = 1024,
    patch_size: int = 16,
    mu: float = 100,
    M: float = 256,
    nb_bins: int = 1024,
    token_shift: int = 32_000,
) -> DatasetDict:
    """
    Load a GIA dataset.

    Example:
        >>> dataset = load_gia_dataset("babyai-go-to")
        >>> dataset[0]["observations/image"].shape
        torch.Size([56, 3, 56, 56])
    """
    dataset = load_dataset("gia-project/gia-dataset", task_name, split="train")
    # TODO: remove this when a better solution is found
    if "babyai" in task_name:
        # remove the "image" column as it is not used
        dataset = dataset.remove_columns("images")

    # Temporarily rename the observations with "observations/" prefix. Maybe we can include this in the dataset later.
    all_keys = dataset.column_names
    observations_keys = [key for key in all_keys if key not in ["actions", "dones", "rewards"]]
    dataset = dataset.rename_columns({key: f"observations/{key}" for key in observations_keys})

    # Convert the dataset to torch tensors
    dataset.set_format(type="numpy")

    # Pack the the dataset into batches of sequences
    # As we change the dataset length, we need to remove all the previous columns.
    batch_generator = BatchGenerator(seq_len, p_prompt, p_end, patch_size, mu, M, nb_bins, token_shift)
    dataset = dataset.map(
        batch_generator,
        batched=True,
        batch_size=10_000,
        remove_columns=dataset.column_names,
    )
    dataset.set_format(type="torch")
    return dataset


if __name__ == "__main__":
    from tqdm import tqdm

    dataset = load_gia_dataset("babyai-go-to")
    daltaloader = DataLoader(dataset.with_format("torch"), batch_size=32)
    for batch in tqdm(daltaloader):
        tqdm.write(str(batch["observations/rgb_images"].shape))
