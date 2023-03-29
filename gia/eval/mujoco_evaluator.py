import os
import random
import numpy as np

import torch
import gymnasium as gym
from tqdm import tqdm
from gia.config.arguments import Arguments, parse_args
from gia.model.gia_model import GiaModel
from gia.datasets.batch_generator import load_prompt_dataset
from gia.processor import MultimodalProcessor

from .evaluator import Evaluator
from .mappings import TASK_TO_ENV_MAPPING, DATASET_FILE_MAPPING


def make_mujoco_env(env_name, render_mode=None):
    return gym.make(env_name, render_mode=render_mode)


class MujocoEvaluator(Evaluator):
    def __init__(self, args: Arguments):
        self.task = "mujoco"
        self.env_names = TASK_TO_ENV_MAPPING[self.task]
        self.data_filepaths = DATASET_FILE_MAPPING[self.task]
        self.args: Arguments = args

    def evaluate(self, model: GiaModel):
        stats = {}
        for env_name, dataset_name in zip(self.env_names, self.data_filepaths):
            stats[env_name] = self._evaluate_env(env_name, dataset_name, model)

    def _evaluate_env(self, env_name, dataset_name, model):
        num_envs = 2
        # number of interactions per sequence. Hard-coded for now
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = GiaModel(Arguments()).to(device)
        env = gym.vector.make(env_name, num_envs)
        num_obs_tokens = env.observation_space.shape[1]
        num_act_tokens = env.action_space.shape[1]
        int_per_seq = self.args.seq_length // (num_obs_tokens + num_act_tokens)

        buffer = {
            "continuous_observations": torch.zeros(
                (num_envs, int_per_seq, num_obs_tokens),
                dtype=torch.long,
                device=device,
            ),
            "continuous_observations_attention_mask": torch.zeros(
                (num_envs, int_per_seq, num_obs_tokens),
                dtype=torch.long,
                device=device,
            ),
            "continuous_actions": torch.zeros(
                (num_envs, int_per_seq, num_act_tokens),
                dtype=torch.long,
                device=device,
            ),
            "continuous_actions_attention_mask": torch.zeros(
                (num_envs, int_per_seq, num_act_tokens),
                dtype=torch.long,
                device=device,
            ),
        }

        prompt_dataset = load_prompt_dataset(dataset_name)
        sampled_prompts_idxs = np.random.randint(0, len(prompt_dataset), size=num_envs)

        # Fill (right side) the buffer with the prompts. Truncate if necessary.
        for key in buffer.keys():
            l = min(buffer[key].shape[1], prompt_dataset[key][sampled_prompts_idxs].shape[1])
            buffer[key][:, -l:] = torch.from_numpy(prompt_dataset[key][sampled_prompts_idxs, -l:]).to(device)

        processor = MultimodalProcessor()

        accum_rewards = np.zeros(num_envs)
        returns = []
        obs, info = env.reset()
        pbar = tqdm(total=self.args.n_episodes)

        while len(returns) < self.args.n_episodes:
            # First, roll the buffer
            for key in buffer.keys():
                buffer[key][:, :-1] = buffer[key][:, 1:]

            # Then, add the last observation to the buffer and mask the last action
            obs_tokens = processor({"continuous_observations": obs})["continuous_observations"]
            buffer["continuous_observations"][:, -1] = torch.from_numpy(obs_tokens).to(device)
            buffer["continuous_actions_attention_mask"][:, -1] = 0

            # Compute the output of the model

            action = np.zeros((num_envs, num_act_tokens))
            token_shift = 32_000

            for i in range(num_act_tokens):
                output = model(buffer, eval=True)
                action_logits = output.logits[:, -num_act_tokens + i, token_shift:]
                action_tokens = torch.argmax(action_logits, -1) + token_shift
                buffer["continuous_actions"][:, -1, i] = action_tokens
                action[:, i] = processor.inverse_tokenize_continuous(action_tokens.cpu()).numpy()
                buffer["continuous_actions_attention_mask"][:, -1, i] = 1

            # TODO: use the output to sample an action
            # action = ...
            # action = env.action_space.sample()

            # Add the action to the buffer and unmask it
            act_tokens = processor({"continuous_actions": action})["continuous_actions"]
            buffer["continuous_actions"][:, -1] = torch.from_numpy(act_tokens).to(device)
            buffer["continuous_actions_attention_mask"][:, -1] = 1

            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            accum_rewards += reward
            for i in range(num_envs):
                if terminated[i] or truncated[i]:
                    returns.append(accum_rewards[i])
                    accum_rewards[i] = 0
                    pbar.update(1)
                    if len(returns) == self.args.n_episodes:
                        break  # in case we finish two episodes at the same time
                    # resample a new prompt and reset masks etc
                    sampled_prompts_idxs = np.random.randint(0, len(prompt_dataset), size=1)

                    # Fill (right side) the buffer with the prompts. Truncate if necessary.
                    for key in buffer.keys():
                        if "loss" in key:  # skip if this key in dict as the model modifies the dict TODO: fix this
                            continue
                        buffer[key][i] *= 0
                        l = min(buffer[key].shape[1], prompt_dataset[key][sampled_prompts_idxs].shape[1])
                        buffer[key][i, :, -l:] = torch.from_numpy(prompt_dataset[key][sampled_prompts_idxs, -l:]).to(
                            device
                        )

        pbar.close()
        env.close()

        return returns
