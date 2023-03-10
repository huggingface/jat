import os
import random

import numpy as np
import torch
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets.core import MultiTaskDataset, TaskDataset
from gia.datasets.mappings import DATASET_FILE_MAPPING
from gia.datasets.utils import tokenize_np


PAD_TOKEN = 10000
END_OF_PROMPT_TOKEN = 10001


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

        assert os.path.exists(dataset_dir)
        dataset = np.load(f"{dataset_dir}/dataset.npy", allow_pickle=True).item()
        obs = dataset["observations"]
        dones = dataset["dones"]
        # rewards = dataset["rewards"]  # unused, for now
        actions = dataset["actions"]

        episode_ends = np.nonzero(dones)[0]
        obs_eps = self.extract_episodes(obs, episode_ends, dones)
        actions_eps = self.extract_episodes(actions, episode_ends, dones)

        self.pack_with_prompting(obs_eps, actions_eps, args.seq_length)

        return

        if use_separator:
            raise NotImplementedError

        if args.use_cache and os.path.exists(
            f"debug_cache/{dataset_dir}/cache.npy",
        ):
            cache = np.load(f"debug_cache/{dataset_dir}/cache.npy", allow_pickle=True).item()
            self.packed_tokens = cache["packed_tokens"]
            self.packed_attn = cache["packed_attn"]
            self.packed_positions = cache["packed_positions"]
            self.packed_loss_masks = cache["packed_loss_masks"]
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

        print(f"Packing sequences for: {self.dataset_dir}")
        (
            self.packed_tokens,
            self.packed_attn,
            self.packed_positions,
            self.packed_loss_masks,
        ) = self.pack(obs_eps, actions_eps, self.seq_len)

        if args.use_cache:
            cache = {
                "packed_tokens": self.packed_tokens,
                "packed_attn": self.packed_attn,
                "packed_positions": self.packed_positions,
                "packed_loss_masks": self.packed_loss_masks,
            }
            os.makedirs(f"debug_cache/{dataset_dir}", exist_ok=True)
            with open(f"debug_cache/{dataset_dir}/cache.npy", "wb") as f:
                np.save(f, cache)

    @staticmethod
    def pack(obs_eps, action_eps, seq_len, overlap=True):
        # TODO: move to utils as this will be used elsewhere
        obs_packs = []  # packed sequences of observations & actions
        attn_packs = []  # packed sequences of indices of where the model can attend to
        position_packs = []  # packed sequences of local positions
        loss_mask_packs = []  # packed sequences of where to mask the loss, only actions should be used in the loss

        obs_len = len(obs_eps[0][0])
        act_len = len(action_eps[0][0])
        obs_act_size = obs_len + act_len
        obs_local_tokens = np.array(list(range(obs_len)))
        action_local_tokens = np.array(list(range(act_len)))
        attn_start = 0

        cur_index = 0
        # TODO find out shape of masks in masked self attn
        cur_obs_pack = []
        cur_attn_pack = []
        cur_pos_pack = []
        cur_loss_mask_pack = []
        for obs_ep, action_ep in tqdm(zip(obs_eps, action_eps)):
            attn_start = cur_index

            for i, (obs, act) in enumerate(zip(obs_ep, action_ep)):
                cur_obs_pack.extend(tokenize_np(obs))
                cur_obs_pack.extend(tokenize_np(act))  # do actions undertake the same tokenization scheme?
                cur_pos_pack.extend(obs_local_tokens)
                cur_pos_pack.extend(action_local_tokens)
                cur_attn_pack.extend([[attn_start, cur_index + obs_len]] * obs_act_size)
                cur_loss_mask_pack.extend([0] * obs_len)
                cur_loss_mask_pack.extend([1] * act_len)
                cur_index += obs_act_size

                if len(cur_obs_pack) > (seq_len - obs_act_size):
                    assert len(cur_obs_pack) == len(cur_attn_pack) == len(cur_pos_pack) == len(cur_pos_pack)
                    # extend with zeros
                    cur_obs_pack.extend([0] * (seq_len - len(cur_obs_pack)))
                    cur_attn_pack.extend([[0, 0]] * (seq_len - len(cur_attn_pack)))
                    cur_pos_pack.extend([0] * (seq_len - len(cur_pos_pack)))
                    cur_loss_mask_pack.extend([0] * (seq_len - len(cur_loss_mask_pack)))

                    obs_packs.append(cur_obs_pack)
                    attn_packs.append(cur_attn_pack)
                    position_packs.append(cur_pos_pack)
                    loss_mask_packs.append(cur_loss_mask_pack)

                    cur_obs_pack = []
                    cur_attn_pack = []
                    cur_pos_pack = []
                    cur_loss_mask_pack = []
                    cur_index = 0
                    attn_start = 0

            if not overlap and len(cur_obs_pack) > 0:  # we dont want overlaps and we didn't just start a new sequence:
                cur_obs_pack.extend([0] * (seq_len - len(cur_obs_pack)))
                cur_attn_pack.extend([[0, 0]] * (seq_len - len(cur_attn_pack)))
                cur_pos_pack.extend([0] * (seq_len - len(cur_pos_pack)))
                cur_loss_mask_pack.extend([0] * (seq_len - len(cur_loss_mask_pack)))

                obs_packs.append(cur_obs_pack)
                attn_packs.append(cur_attn_pack)
                position_packs.append(cur_pos_pack)
                loss_mask_packs.append(cur_loss_mask_pack)

                cur_obs_pack = []
                cur_attn_pack = []
                cur_pos_pack = []
                cur_loss_mask_pack = []
                cur_index = 0
                attn_start = 0

        return (
            np.array(obs_packs),
            np.array(attn_packs),
            np.array(position_packs),
            np.array(loss_mask_packs),
        )

    @staticmethod
    def sample(p):
        return random.random() < p

    @staticmethod
    def pack_with_prompting(obs_eps, action_eps, seq_len, p_prompt=0.25, p_end=0.5):
        obs_len = len(obs_eps[0][0])
        act_len = len(action_eps[0][0])
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

        packed_tokens = []
        for tokenized_episode in tokenized_episodes:
            i = 0
            while i < len(tokenized_episode):
                tokens = []
                required_tokens = (seq_len // obs_act_size) * obs_act_size
                if random.random() < p_prompt:
                    # we are sampling a prompt from a random episode and prepending it
                    episode_index = random.randrange(0, len(tokenized_episodes))
                    random_episode = tokenized_episodes[episode_index]
                    n_tokens = random.randrange(obs_act_size, min(required_tokens, len(random_episode)), obs_act_size)
                    required_tokens -= n_tokens
                    if random.random() < p_end:
                        # the prompt is taken from the end of a random episode
                        tokens.extend(random_episode[-n_tokens:])  # n tokens from the end of the episode
                    else:
                        # the prompt is uniformly taken somewhere in the episode
                        start = random.randrange(0, len(random_episode) - n_tokens, obs_act_size)
                        # n_tokens uniformly sampled from ep
                        tokens.extend(random_episode[start : start + n_tokens])
                    tokens.append(END_OF_PROMPT_TOKEN)

                # take all the required tokens, or whatever is left in the episode
                m = min(required_tokens, len(tokenized_episode[i:]))
                assert m % obs_act_size == 0
                # you can take just required_tokens, as the overflow is truncated
                tokens.extend(tokenized_episode[i : i + m])
                i += m

                # pad the takens
                tokens.extend([PAD_TOKEN] * (seq_len - len(tokens)))
                assert len(tokens) == seq_len
                packed_tokens.append(tokens)

            assert i == len(tokenized_episode)  # we should have appended all tokens

        # print(packed_tokens)

    @staticmethod
    def extract_episodes(data, episode_ends, dones):
        episodes = []
        index = 0
        for end in episode_ends[:20]:
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
            "attn_ids": self.packed_attn[idx],
            "local_position_ids": self.packed_positions[idx],
            "loss_masks": self.packed_loss_masks[idx],
        }


if __name__ == "__main__":

    for sl in [75, 100, 150, 200, 171]:

        args = Arguments()
        args.seq_length = sl
        dataset = MujocoTaskDataset(args, "data/imitation/mujoco/prj_gia_dataset_mujoco_ant_1111")
