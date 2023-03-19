import random

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from gia.tokenizers.multimodal_tokenizer import MultiModalTokenizer


class _SingleTaskDataset(Dataset):
    """
    Dataset for the GIA project.
    """

    def __init__(self, p_prompt=0.25, p_end=0.5, seq_len=512):
        self.separator_token = 32_000
        self.p_prompt = p_prompt
        self.p_end = p_end
        self.seq_len = seq_len
        dataset = load_dataset("gia-project/gia-dataset", "babyai-go-to", split="train")
        all_keys = dataset.column_names
        observations_keys = [key for key in all_keys if key not in ["actions", "dones", "rewards"]]
        # Temporarily rename the observations with "observations/" prefix. Maybe we can include this in the dataset later.
        dataset = dataset.rename_columns({key: f"observations/{key}" for key in observations_keys})
        dataset.set_format("torch")
        tokenizer = MultiModalTokenizer()
        dataset = dataset.map(tokenizer, batched=True)
        # As we change the dataset length, we need to remove all the previous columns.
        self.dataset = dataset.map(
            self.pack_with_prompting, batched=True, batch_size=-1, remove_columns=dataset.column_names
        )

    @staticmethod
    def flatten_dataset(dataset, observation_keys, separator_token):
        """
        Flatten the input dataset into a single sequence of tokens, loss masks,
        and episode done flags.

        Args:
            dataset (dict): A dictionary containing the input data, including
                observation and action tokens and episode done flags.
            observation_keys (list): A list of strings containing the names of the
                observation keys in the dataset.
            separator_token (str): A string representing the separator token to use
                between episodes.

        Returns:
            A tuple containing the flattened tokens, flattened loss masks, and
            episode done flags as lists.

        Example:
            >>> dataset = {
            ...    "obs1_tokens": [[1, 2], [6, 7], [12]],
            ...    "obs2_tokens": [[3], [8, 9], [13]],
            ...    "actions_tokens": [[4, 5], [10, 11], [14, 15]],
            ...    "dones": [False, False, True]
            ... }
            >>> observation_keys = ["obs1_tokens", "obs2_tokens"]
            >>> separator_token = 0
            >>> flat_tokens, flat_loss_masks, episode_done_flags = flatten_dataset(
            ...     dataset, observation_keys, separator_token
            ... )
            >>> flat_tokens
            [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 0, 14, 15]
            >>> flat_loss_masks
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
            >>> episode_done_flags
            [False, False, False, False, False, False, False, False, False, False, False, True, False, False, False]
        """
        flat_tokens = []
        flat_loss_masks = []
        episode_done_flags = []

        for idx in range(len(dataset["actions_tokens"])):
            episode_done_flags.append(dataset["dones"][idx])

            for obs_key in observation_keys:
                flat_tokens.extend(dataset[obs_key][idx])
                flat_loss_masks.extend([0] * len(dataset[obs_key][idx]))
                episode_done_flags.extend([False] * (len(dataset[obs_key][idx]) - 1))

            flat_tokens.append(separator_token)
            flat_loss_masks.append(0)
            episode_done_flags.append(False)

            flat_tokens.extend(dataset["actions_tokens"][idx])
            flat_loss_masks.extend([1] * len(dataset["actions_tokens"][idx]))
            episode_done_flags.extend([False] * len(dataset["actions_tokens"][idx]))
        # Convert to torch tensors
        flat_tokens = torch.tensor(flat_tokens)
        flat_loss_masks = torch.tensor(flat_loss_masks)
        episode_done_flags = torch.tensor(episode_done_flags)
        return flat_tokens, flat_loss_masks, episode_done_flags

    @staticmethod
    def generate_prompted_sequences(episode_done_flags, p_prompt, p_end, seq_len):
        """
        Generate prompted sequences from the flattened token and loss mask lists, using a specified
        prompt probability and prompt end probability.

        Args:
            episode_done_flags (list): A list of episode done flags, indicating the end of each episode
                in the flattened token and loss mask lists.
            p_prompt (float): The probability of generating a prompted sequence.
            p_end (float): The probability of taking the prompt from the end of an episode, rather than
                uniformly within the episode.
            seq_len (int): The desired length of each generated sequence, including any generated prompt.

        Returns:
            A list of indexes corresponding to the generated sequences.

        Example:
            >>> # All sequences are prompted with the end of the an episode.
            >>> generate_prompted_sequences(
            ...     episode_done_flags=[False, False, False, False, False, False, False, True, False],
            ...     p_prompt=1.0,
            ...     p_end=1.0,
            ...     seq_len=4
            ... )
            [[5, 6, 0, 1], [5, 6, 2, 3], [5, 6, 4, 5], [6, 6, 7, 8]]

            >>> # None of the sequences are prompted.
            >>> generate_prompted_sequences(
            ...     episode_done_flags=[False, False, False, False, False, False, False, True, False],
            ...     p_prompt=0.0,
            ...     p_end=1.0,
            ...     seq_len=4
            ... )
            [[0, 1, 2, 3], [4, 5, 6, 7]]

            >>> # Some sequences are prompted, but not all (here, just the last one)
            >>> generate_prompted_sequences(
            ...     episode_done_flags=[False, False, False, False, False, False, False, True, False],
            ...     p_prompt=0.5,
            ...     p_end=0.5,
            ...     seq_len=4
            ... )
            [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 8]]
        """
        dataset_indexes = []
        episode_start_idxs = [idx for idx, done in enumerate(episode_done_flags) if done]
        current_ep_start = 0

        while True:
            if random.random() < p_prompt:
                num_prompt_tokens = random.randint(0, seq_len - 1)
                episode_end_idx = random.choice(episode_start_idxs)

                if random.random() < p_end:
                    prompt_indexes = list(range(episode_end_idx - num_prompt_tokens, episode_end_idx))
                else:
                    prompt_episode_start = random.randint(0, len(episode_done_flags) - num_prompt_tokens)
                    prompt_indexes = list(range(prompt_episode_start, prompt_episode_start + num_prompt_tokens))
            else:
                num_prompt_tokens = 0
                prompt_indexes = []

            num_tokens = seq_len - num_prompt_tokens

            if current_ep_start + num_tokens > len(episode_done_flags):
                break

            indexes = prompt_indexes + list(range(current_ep_start, current_ep_start + num_tokens))
            dataset_indexes.append(indexes)
            current_ep_start += num_tokens
        # Convert to torch tensors
        return torch.tensor(dataset_indexes)

    def pack_with_prompting(self, dataset):
        # First, get the total numbers of tokens in the dataset
        observation_keys = [
            key for key in dataset.keys() if key.startswith("observations/") and key.endswith("_tokens")
        ]

        # Then, we need to flatten the dataset
        flat_tokens, flat_loss_masks, episode_done_flags = self.flatten_dataset(
            dataset, observation_keys, self.separator_token
        )
        batch_idxs = self.generate_prompted_sequences(episode_done_flags, self.p_prompt, self.p_end, self.seq_len)
        return {
            "tokens": flat_tokens[batch_idxs],
            "loss_masks": flat_loss_masks[batch_idxs],
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    dataset = _SingleTaskDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(loader)))
