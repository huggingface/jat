import random
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from gia.tokenizers.multimodal_tokenizer import MultiModalTokenizer
from gia.utils.utils import to_channel_first

ACTION_POSITION = 999


def is_image(x: Any) -> bool:
    """
    Check if input is an image.

    Returns True if the input is a torch tensor with 4 dimensions.
    """
    return isinstance(x, Tensor) and x.dim() == 4  # shape (batch_size, channels, height, width)


class GiaDataset(Dataset):
    """
    Dataset for the GIA project.
    """

    def __init__(self, p_prompt=0.25, p_end=0.5, seq_len=512):
        self.separator_token = 32_000
        self.p_prompt = p_prompt
        self.p_end = p_end
        self.seq_len = seq_len
        self.patch_size = 16
        dataset = load_dataset("gia-project/gia-dataset", "babyai-go-to", split="train")
        # Remove the "image" column as it is not used
        dataset = dataset.remove_columns("images")
        # Temporarily rename the observations with "observations/" prefix. Maybe we can include this in the dataset later.
        all_keys = dataset.column_names
        observations_keys = [key for key in all_keys if key not in ["actions", "dones", "rewards"]]
        dataset = dataset.rename_columns({key: f"observations/{key}" for key in observations_keys})
        # Convert the dataset to torch tensors
        dataset.set_format("torch")
        # Tokenize the dataset
        tokenizer = MultiModalTokenizer()
        dataset = dataset.map(tokenizer, batched=True, load_from_cache_file=False)
        # Pack the the dataset into batches of sequences
        # As we change the dataset length, we need to remove all the previous columns.
        self.dataset = dataset.map(
            self.pack_and_prompt, batched=True, batch_size=-1, remove_columns=dataset.column_names
        )

    @staticmethod
    def get_num_patches(x, patch_size):
        # x is a batch of images
        _, _, height, width = x.shape
        return (height // patch_size) * (width // patch_size)

    def pack_and_prompt(self, dataset):
        # Compute the number of embeddings needed for the observations and actions

        # First get the keys for the observations. Caution, images are handled separately since they are not tokenized
        observation_keys = [
            key for key in dataset.keys() if key.startswith("observations/") and key.endswith("_tokens")
        ]
        # Concatenate the observations and actions, and ped when necessary
        observation_tokens = {key: pad_sequence(dataset[key], batch_first=True) for key in observation_keys}
        num_obs_tokens = sum(tokens.size(1) for tokens in observation_tokens.values())

        # Special case for images: we need to deduce the number of embeddings from the image size
        # `image_keys` and `observation_keys` are intended not to overlap since the images are not tokenized
        image_keys = [key for key in dataset.keys() if key.startswith("observations/") and is_image(dataset[key])]
        # First, make sure that all images are in channels first format
        for key in image_keys:
            dataset[key] = to_channel_first(dataset[key])
        # The number of embeddings is the sum of the number of patches for each image
        num_image_patches = sum(self.get_num_patches(dataset[key], self.patch_size) for key in image_keys)
        num_emb_per_observation = num_obs_tokens + num_image_patches

        # Do the same for the actions (here it's simpler since there is only one key, and the action is not an image)
        action_tokens = pad_sequence(dataset["actions_tokens"], batch_first=True)
        num_emb_per_action = action_tokens.size(1)

        # Set the position for observations and actions
        # observation_indices = torch.arange(num_emb_per_observation)
        # action_indices = torch.ones(num_emb_per_action, dtype=torch.int64) * ACTION_POSITION

        # obs_act_pos_indices = list(range(num_emb_per_observation)) + list(range(num_emb_per_action))
        # obs_act_loss_mask = [0] * num_emb_per_observation + [1] * num_emb_per_action

        num_emb_per_interaction = num_emb_per_observation + num_emb_per_action
        num_interractions_per_seq = self.seq_len // num_emb_per_interaction
        # assert num_emb_per_interaction * 2 + 1 <= self.seq_len  # we require at least two obs_acts in the sequence + prompt token

        # # tokenize all the episodes
        # tokenized_episodes = []
        # for obs_ep, action_ep in tqdm(zip(obs_eps, action_eps)):
        #     cur_ep_tokens = []
        #     for obs, act in zip(obs_ep, action_ep):
        #         cur_ep_tokens.extend(tokenize_np(obs))
        #         cur_ep_tokens.extend(tokenize_np(act))
        #     tokenized_episodes.append(cur_ep_tokens)

        # From the done flags, compute the indexes of the start of each episode
        episode_end_idxs = [i for i, done in enumerate(dataset["dones"]) if done]
        current_ep_start = 0
        while True:
            if random.random() < self.p_prompt:
                num_prompt_interactions = random.randint(1, num_interractions_per_seq)
                num_interractions = num_interractions_per_seq - num_prompt_interactions
                prompt_episode_idx = random.randint(0, len(episode_end_idxs) - 1)
                if random.random() < self.p_end:  # Prompt at the end of the episode
                    # Ensure that you don't take from the previous episode
                    previous_episode_end_idx = (
                        episode_end_idxs[prompt_episode_idx - 1] if prompt_episode_idx > 0 else -1
                    )
                    episode_end_idx = episode_end_idxs[prompt_episode_idx]
                    prompt_episode_length = episode_end_idx - previous_episode_end_idx
                    num_prompt_interactions = min(num_prompt_interactions, prompt_episode_length)
                    prompt_indexes = list(range(episode_end_idx - num_prompt_interactions + 1, episode_end_idx + 1))
                else:  # Prompt anywhere in the episode
                    # Get the range for the possible start of the prompt
                    previous_episode_end_idx = (
                        episode_end_idxs[prompt_episode_idx - 1] if prompt_episode_idx > 0 else -1
                    )
                    episode_end_idx = episode_end_idxs[prompt_episode_idx]
                    prompt_episode_length = episode_end_idx - previous_episode_end_idx
                    num_prompt_interactions = min(num_prompt_interactions, prompt_episode_length)
                    prompt_start_idx = random.randint(
                        previous_episode_end_idx + 1, episode_end_idx - num_prompt_interactions + 1
                    )
                    prompt_indexes = list(range(prompt_start_idx, prompt_start_idx + num_prompt_interactions))
            else:
                num_prompt_interactions = 0
                num_interractions = num_interractions_per_seq
                prompt_indexes = []

            if current_ep_start + num_interractions > len(dataset["dones"]):
                break
            episode_indexes = list(range(current_ep_start, current_ep_start + num_interractions))

            indexes = prompt_indexes + episode_indexes
            observations = {key: [val[idx] for idx in indexes] for key, val in observation_tokens.items()}
            images = {key: [dataset[idx] for idx in indexes] for key in image_keys}
            actions = action_tokens[indexes]

            current_ep_start += num_interractions

        return {
            **observations,
            **images,
            "actions_tokens": actions,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    dataset = GiaDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(loader)))
