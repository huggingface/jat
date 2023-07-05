import gymnasium as gym
import pytest

from gia.eval.rl.envs.babyai import BABYAI_ENV_NAMES, BabyAIWrapper


@pytest.mark.parametrize("env_name", BABYAI_ENV_NAMES)
def test_babyai_wrapper(env_name):
    env = gym.make(env_name)
    env = BabyAIWrapper(env)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()
