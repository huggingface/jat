import os
import random

import numpy as np
import torch
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets.core import MultiTaskDataset, TaskDataset
from gia.datasets.mappings import DATASET_FILE_MAPPING
from gia.datasets.utils import tokenize_np

# TODO: these should change depending on vocal size etc
PAD_TOKEN = 1024
END_OF_PROMPT_TOKEN = 1025
END_OF_PROMPT_POSITION_TOKEN = 30
POSITION_PAD_TOKEN = 31


class MujocoDataset(MultiTaskDataset):
    def __init__(self, task: str, args: Arguments):
        super().__init__()
        self.task = task
        dataset_dirs = DATASET_FILE_MAPPING[task]
        self.task_datasets = [MujocoTaskDataset(args, dataset_dir) for dataset_dir in dataset_dirs]
        self.dataset_len = sum(len(d) for d in self.task_datasets)


class MujocoTaskDataset(TaskDataset):
    def __init__(self, args: Arguments, dataset_dir: str, use_separator: bool = False) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.seq_len = args.seq_length

        if use_separator:
            raise NotImplementedError

        if args.use_cache and os.path.exists(
            f"debug_cache/{dataset_dir}/cache.npy",
        ):
            cache = np.load(f"debug_cache/{dataset_dir}/cache.npy", allow_pickle=True).item()
            self.packed_tokens = cache["packed_tokens"]
            self.packed_positions = cache["packed_positions"]
            self.packed_loss_masks = cache["packed_loss_masks"]
            self.packed_attn_masks = cache["packed_attn_masks"]
            return

        assert os.path.exists(dataset_dir)
        dataset = np.load(f"{dataset_dir}/dataset.npy", allow_pickle=True).item()
        obs = dataset["observations"]
        dones = dataset["dones"]
        # rewards = dataset["rewards"]  # unused, for now
        actions = dataset["actions"]

        episode_ends = np.nonzero(dones)[0]
        obs_eps = self.extract_episodes(obs, episode_ends, dones)
        actions_eps = self.extract_episodes(actions, episode_ends, dones)

        (
            self.packed_tokens,
            self.packed_positions,
            self.packed_loss_masks,
            self.packed_attn_masks,
        ) = self.pack_with_prompting(obs_eps, actions_eps, args.seq_length)

        if args.use_cache:
            cache = {
                "packed_tokens": self.packed_tokens,
                "packed_positions": self.packed_positions,
                "packed_loss_masks": self.packed_loss_masks,
                "packed_attn_masks": self.packed_attn_masks,
            }
            os.makedirs(f"debug_cache/{dataset_dir}", exist_ok=True)
            with open(f"debug_cache/{dataset_dir}/cache.npy", "wb") as f:
                np.save(f, cache)

    @staticmethod
    def pack_with_prompting(obs_eps, action_eps, seq_len, p_prompt=0.25, p_end=0.5):
        obs_len = len(obs_eps[0][0])
        act_len = len(action_eps[0][0])

        obs_act_pos_indices = list(range(obs_len)) + list(range(act_len))
        obs_act_loss_mask = [0] * obs_len + [1] * act_len

        obs_act_size = obs_len + act_len
        assert obs_act_size * 2 + 1 <= seq_len  # we require at least two obs_acts in the sequence + prompt token

        # tokenize all the episodes
        tokenized_episodes = []
        for obs_ep, action_ep in tqdm(zip(obs_eps, action_eps)):
            cur_ep_tokens = []
            for obs, act in zip(obs_ep, action_ep):
                cur_ep_tokens.extend(tokenize_np(obs))
                cur_ep_tokens.extend(tokenize_np(act))
            tokenized_episodes.append(cur_ep_tokens)

        token_seqs = []
        pos_seqs = []
        loss_mask_seqs = []
        pad_mask_seqs = []
        for tokenized_episode in tokenized_episodes:
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
        return len(self.packed_tokens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #

        return {
            "task": self.dataset_dir,
            "tokens": self.packed_tokens[idx],
            "attn_mask": self.packed_attn_masks[idx],
            "local_position_ids": self.packed_positions[idx],
            "loss_mask": self.packed_loss_masks[idx],
        }


if __name__ == "__main__":

    for sl in [75, 100, 150, 200, 171, 256, 512, 1024, 2048]:
        args = Arguments()
        args.use_cache = False
        args.seq_length = sl
        dataset = MujocoTaskDataset(args, "data/imitation/mujoco/prj_gia_dataset_mujoco_ant_1111")
