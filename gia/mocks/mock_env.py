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
        obss = []
        rewards = []
        dones = []
        infos = []

        for action, env in zip(actions, self.envs):
            obs, reward, done, info = env.step(action)
            obss.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        obss = lod_to_dol(obss)
        for k, v in obss.items():
            obss[k] = np.array(v)
        rewards = np.array(rewards)
        dones = np.array(dones)

        return obs, rewards, dones, infos

    def reset(self):
        obss = []
        for env in self.envs:
            obs = env.reset()
            obss.append(obs)

        obss = lod_to_dol(obss)
        for k, v in obss.items():
            obss[k] = np.array(v)
        return obss


class MockImageEnv(Env):
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
        action_dim: int = 6,
        screen_height: int = 84,
        screen_width: int = 84,
        n_channels: int = 1,
        discrete: bool = True,
        channel_first: bool = True,
    ):
        self.observation_space = (screen_height, screen_width, n_channels)
        if channel_first:
            self.observation_space = (n_channels, screen_height, screen_width)
        self.observation_space = spaces.Dict(
            vec=spaces.Box(low=0, high=255, shape=self.observation_space, dtype=np.uint8)
        )
        if discrete:
            self.action_space = spaces.Dict(action=spaces.Discrete(action_dim))
        else:
            self.action_space = spaces.Dict(action=spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32))
        self.ep_length = 10
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        return self.observation_space.sample()

    def step(self, action: Union[np.ndarray, int]):
        reward = 0.0
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, done, {}

    def render(self, mode: str = "human") -> None:
        pass
