from collections import defaultdict
from typing import Optional, Union

import numpy as np
from gym import Env, Space
from gym import spaces

from gia.utils.utils import dol_to_lod, lod_to_dol


class MockBatchedEnv:
    def __init__(self, env_fn, num_parallel) -> None:
        self.envs = []
        self.num_parallel = num_parallel
        for i in range(num_parallel):
            self.envs.append(env_fn())

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    def sample_actions(self):
        actions = [self.action_space.sample() for _ in range(self.num_parallel)]
        actions = lod_to_dol(actions)
        for k, v in actions.items():
            actions[k] = np.array(v)
        return actions

    def step(self, actions):

        actions = dol_to_lod(actions)
        assert len(actions) == len(self.envs)
        obss = []
        rewards = []
        terms = []
        truncs = []
        infos = []

        for action, env in zip(actions, self.envs):
            obs, reward, term, trunc, info = env.step(action)
            obss.append(obs)
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)

        obss = lod_to_dol(obss)
        for k, v in obss.items():
            obss[k] = np.array(v)
        rewards = np.array(rewards)
        terms = np.array(terms)
        truncs = np.array(truncs)

        return obss, rewards, terms, truncs, infos

    def reset(self):
        obss = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()

            obss.append(obs)
            infos.append(info)

        obss = lod_to_dol(obss)
        for k, v in obss.items():
            obss[k] = np.array(v)
        return obss, infos


class MockEnv(Env):
    """
    Mock image environment for testing purposes, it mimics Atari games.
    :param action_dim: Number of discrete actions
    :param screen_height: Height of the image
    :param screen_width: Width of the image
    :param n_channels: Number of color channels
    :param discrete: Create discrete action space instead of continuous
    :param channel_first: Put channels on first axis instead of last
    """

    def __init__(
        self,
        observation_space,
        action_space,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.ep_length = 10
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self.observation_space.sample(), {}

    def step(self, action: Union[np.ndarray, int]):
        reward = 0.0
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, done, done, {}

    def render(self, mode: str = "human") -> None:
        pass
