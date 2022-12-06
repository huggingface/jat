import pytest
import numpy as np

from gia.config.config import Config
from gia.envs.atari import make_atari_env
from gia.envs.wrappers import autowrap


@pytest.mark.parametrize(
    "env_name",
    [
        "atari_beamrider",
        "atari_breakout",
        "atari_montezuma",
        "atari_pong",
        "atari_qbert",
    ],
)
def test_autowrap(env_name):
    config = Config.build()
    env = make_atari_env(env_name, config, None)
    env = autowrap(env)
    obs, info = env.reset()
    assert obs["obs"].shape == (config.envs.agents_per_env, 4, 84, 84)
    assert env.num_envs == config.envs.agents_per_env
    for i in range(10):
        actions = np.array([env.action_space.sample() for _ in range(config.envs.agents_per_env)])
        obs, reward, term, trunc, info = env.step(actions)
        assert obs["obs"].shape == (config.envs.agents_per_env, 4, 84, 84)
        assert reward.shape == (config.envs.agents_per_env,)
        assert term.shape == (config.envs.agents_per_env,)
        assert trunc.shape == (config.envs.agents_per_env,)
