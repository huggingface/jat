import numpy as np
import gym


class EnvPoolResetFixWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        needs_reset = np.nonzero(terminated | truncated)[0]
        obs[needs_reset], _ = self.env.reset(needs_reset)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        kwargs.pop("seed", None)  # envpool does not support the seed in reset, even with the updated API
        return super().reset(**kwargs)


class BatchedRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, num_envs, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", num_envs)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, infos = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - terminated
        self.episode_lengths *= 1 - terminated
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, terminated, truncated, infos


class DictObservationsWrapper(gym.Wrapper):
    """Guarantees that the environment returns observations as dictionaries of lists (batches)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(dict(obs=self.observation_space))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return dict(obs=obs), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return dict(obs=obs), rew, terminated, truncated, info


class DictActionsWrapper(gym.Wrapper):
    """Guarantees that the environment accepts actions as dictionaries of lists (batches)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space: gym.spaces.Dict = gym.spaces.Dict(dict(actions=self.action_space))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action["actions"])
        return obs, rew, terminated, truncated, info


def autowrap(env):
    # adds wrappers to an env so if complies with the lib
    # final envs should have:
    # - Flat Dict Observation space - DONE
    # - Flat Dict Action space - DONE
    # - Return Torch Tensors for observations
    # - Be in batch mode by default

    if isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Discrete)):
        env = DictObservationsWrapper(env)
    elif isinstance(env.observation_space, gym.spaces.Tuple):
        raise NotImplementedError

    if isinstance(env.action_space, (gym.spaces.Box, gym.spaces.Discrete)):
        env = DictActionsWrapper(env)
    elif isinstance(env.action_space, gym.spaces.Tuple):
        raise NotImplementedError

    return env
