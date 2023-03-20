import os
import random
import numpy as np

import torch
import gymnasium as gym

from gia.config.arguments import Arguments, parse_args
from gia.datasets.mappings import DATASET_FILE_MAPPING
from gia.datasets.mujoco_dataset import MujocoTaskDataset
from gia.datasets.utils import tokenize_np, inverse_tokenize_np
from gia.model.gia_model import GiaModel

from gia.tokenizers.constants import (
    PAD_TOKEN,
    END_OF_PROMPT_POSITION_TOKEN,
    POSITION_PAD_TOKEN,
    END_OF_PROMPT_TOKEN,
)

from .evaluator import Evaluator
from .mappings import TASK_TO_ENV_MAPPING


def make_mujoco_env(env_name, render_mode=None):
    return gym.make(env_name, render_mode=render_mode)


class MujocoEvaluator(Evaluator):
    def __init__(self, args: Arguments):
        self.env_names = TASK_TO_ENV_MAPPING["mujoco"]
        self.data_filepaths = DATASET_FILE_MAPPING["mujoco"]
        self.args: Arguments = args
        self.tokenizer = tokenize_np  # TODO: replace with Quentin's tokenizer
        self.inv_tokenizer = inverse_tokenize_np

    def evaluate(self, model):
        stats = {}
        for env_name, dataset_dir in zip(self.env_names, self.data_filepaths):
            stats[env_name] = self._evaluate_env(env_name, dataset_dir, model)

    def _evaluate_env(self, env_name, dataset_dir, model):
        tokenized_episodes, obs_len, act_len = MujocoTaskDataset.load_and_tokenize(dataset_dir)
        obs_indices = range(obs_len)
        act_indices = range(act_len)
        env = make_mujoco_env(env_name)

        returns = []
        for i in range(self.args.n_episodes):
            print(i)
            rewards = []
            tokens, pos_tokens = self._sample_random_prompt(tokenized_episodes, obs_len, act_len)
            obs, info = env.reset()
            tokens.extend(self.tokenizer(obs))
            pos_tokens.extend(obs_indices)

            done = False
            while not done:
                action = []
                # This whole loop is probably really inefficient and can be optimized a lot
                for i in range(act_len):
                    tokens = tokens[-self.args.seq_length :]
                    pos_tokens = pos_tokens[-self.args.seq_length :]
                    # Optimization to make, use kv cache
                    action_token = model.predict_next_token(
                        {  # TODO: is padding /masking required here?
                            "tasks": ["mujoco"],
                            "tokens": torch.LongTensor(tokens).to(model.device).unsqueeze(0),
                            "local_position_ids": torch.LongTensor(pos_tokens).to(model.device).unsqueeze(0),
                        }
                    )
                    action.append(action_token.item())
                    tokens.append(action_token.item())
                    pos_tokens.append(act_indices[i])  # we could append i, but perhaps this indices will change?

                action = self.inv_tokenizer(np.array(action))
                obs, reward, term, truc, info = env.step(action)
                done = term or truc

                rewards.append(reward)

            returns.append(sum(rewards))

        return returns

    def _sample_random_prompt(self, tokenized_episodes, obs_len, act_len):
        obs_act_size = obs_len + act_len
        obs_act_pos_indices = list(range(obs_len)) + list(range(act_len))
        max_length = ((self.args.seq_length - 1 - obs_len) // (obs_act_size)) * obs_act_size  # EOP TOKEN
        n_eps = len(tokenized_episodes)

        tokenized_episode = tokenized_episodes[random.randrange(0, n_eps)]
        tries = 100  # it may be the all eps are shorter than 1024
        while tries and len(tokenized_episode) < obs_act_size:
            tokenized_episode = tokenized_episodes[random.randrange(0, n_eps)]
            tries -= 1

        assert len(tokenized_episode) >= max_length  # this will fail in a few cases, let's find out which envs
        prompt_tokens = tokenized_episode[:max_length] + [END_OF_PROMPT_TOKEN]  # Should this be from the end of an ep?
        position_tokens = obs_act_pos_indices * ((self.args.seq_length - 1 - obs_len) // obs_act_size)
        position_tokens.append(END_OF_PROMPT_POSITION_TOKEN)

        assert len(prompt_tokens) == len(position_tokens)

        return prompt_tokens, position_tokens
