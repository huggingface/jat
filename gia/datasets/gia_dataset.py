from datasets import load_dataset
from gia.tokenizers.multimodal_tokenizer import MultiModalTokenizer
import torch
import random


def get_episode_idx(dones):
    """
    Get episode indices.

    Example:
        >>> dones = [False, False, True, False, False, False, True]
        >>> get_episode_idx(dones)
        [[0, 1, 2], [3, 4, 5, 6]]
    Args:
        dones (list): List of dones.

    Returns:
        list: List of episode indices.
    """
    episode_idx = []
    episode = []
    for idx, done in enumerate(dones):
        episode.append(idx)
        if done:
            episode_idx.append(episode)
            episode = []
    return episode_idx


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer = MultiModalTokenizer()
        self.dataset = load_dataset("gia-project/gia-dataset", "babyai-go-to", split="train")
        all_keys = self.dataset.column_names
        self.observations_keys = [key for key in all_keys if key not in ["actions", "dones", "rewards"]]
        self.dataset.set_format("torch")
        self.dataset = self.dataset.map(self.tokenizer, batched=True, load_from_cache_file=False)
        self.dataset = self.dataset.map(self.pack_with_prompting, batched=True, batch_size=-1)

    def pack_with_prompting(self, dataset, p_prompt=0.25, p_end=0.5, seq_len=512):
        episode_end = dataset["dones"].nonzero(as_tuple=False).squeeze(-1)  # Example: [2, 6, 10, 22]
        a = [dataset[f"{obs_key}_tokens"] for obs_key in self.observations_keys]
        total_number_of_tokens = sum(len(obs_token) for obs_token in dataset["obs_tokens"])
        current_ep_start = 0
        while True:
            if random.random() < p_prompt:
                # Sequence is prepended with a prompt
                n_tokens = random.randint(0, seq_len - 1)
                n_prompt_tokens = seq_len - n_tokens
                if random.random() < p_end:
                    # The prompt is taken from the end of a random episode
                    prompt_episode_end = random.choice(episode_end)
                else:
                    # The prompt is uniformly taken somewhere in the episode
                    prompt_episode_end = random.randint(0, len(dataset) - n_tokens)
            else:
                # Sequence is not prepended with a prompt
                n_tokens = seq_len
                prompt_idxs = []
            
            # First, create the prompt
            
            # The sequence is taken from the current episode
            batch_idxs = list(range(current_ep_start, current_ep_start + n_tokens))



        obs_eps = dataset["obs"]
        action_eps = dataset["action"]
        seq_len = 512

        p_end = 0.5
        obs_len = len(obs_eps[0][0])
        act_len = len(action_eps[0][0])

        obs_act_pos_indices = list(range(obs_len)) + list(range(act_len))
        obs_act_loss_mask = [0] * obs_len + [1] * act_len

        obs_act_size = obs_len + act_len
        assert obs_act_size * 2 + 1 <= seq_len  # we require at least two obs_acts in the sequence + prompt token

        # tokenized_episodes = [
        #   [token11, token12, ...], # episode 1
        #   [token21, token22, ...], # episode 1
        # ...
        # ]

        token_seqs = []
        pos_seqs = []
        loss_mask_seqs = []
        pad_mask_seqs = []
        for tokenized_episode in range(len(episode_idxs)):
            # tokenized_episode = [token11, token12, ...]
            i = 0
            while i < len(tokenized_episode):
                tokens = []
                positions = []
                loss_masks = []
                required_tokens = (seq_len // obs_act_size) * obs_act_size
                if random.random() < p_prompt:
                    # we are sampling a prompt from a random episode and prepending it
                    episode_index = random.randrange(0, len(tokenized_episodes))
                    random_episode = tokenized_episodes[episode_index]
                    n_tokens = random.randrange(obs_act_size, min(required_tokens, len(random_episode)), obs_act_size)
                    required_tokens -= n_tokens

                    positions.extend(obs_act_pos_indices * (n_tokens // obs_act_size))
                    loss_masks.extend([0] * n_tokens)

                    if random.random() < p_end:
                        # the prompt is taken from the end of a random episode
                        tokens.extend(random_episode[-n_tokens:])  # n tokens from the end of the episode
                    else:
                        # the prompt is uniformly taken somewhere in the episode
                        start = random.randrange(0, len(random_episode) - n_tokens, obs_act_size)
                        # n_tokens uniformly sampled from ep
                        tokens.extend(random_episode[start : start + n_tokens])

                    tokens.append(END_OF_PROMPT_TOKEN)
                    positions.append(END_OF_PROMPT_POSITION_TOKEN)
                    loss_masks.append(0)

                # take all the required tokens, or whatever is left in the episode
                m = min(required_tokens, len(tokenized_episode[i:]))
                assert m % obs_act_size == 0
                # you can take just required_tokens, as the overflow is truncated
                tokens.extend(tokenized_episode[i : i + m])
                positions.extend(obs_act_pos_indices * (m // obs_act_size))
                loss_masks.extend(obs_act_loss_mask * (m // obs_act_size))

                i += m

                # pad the takens
                n_pad = seq_len - len(tokens)

                pad_mask = [1] * len(tokens) + [0] * n_pad
                tokens.extend([PAD_TOKEN] * n_pad)
                positions.extend([POSITION_PAD_TOKEN] * n_pad)
                loss_masks.extend([0] * n_pad)

                assert seq_len == len(tokens) == len(positions) == len(loss_masks) == len(pad_mask)
                token_seqs.append(tokens)
                pos_seqs.append(positions)
                loss_mask_seqs.append(loss_masks)
                pad_mask_seqs.append(pad_mask)

            assert i == len(tokenized_episode)  # we should have appended all tokens

        return (
            np.array(token_seqs),
            np.array(pos_seqs),
            np.array(loss_mask_seqs),
            np.array(pad_mask_seqs),
        )

    @staticmethod
    def extract_episodes(data, episode_ends, dones):
        episodes = []
        index = 0
        for end in episode_ends:
            episodes.append(data[index:end])
            index = end
            assert dones[end] == 1

        return episodes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        return x


if __name__ == "__main__":
    dataset = MyDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for x in loader:
        print(x.shape)
