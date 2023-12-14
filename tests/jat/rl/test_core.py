import numpy as np
import pytest

from jat.eval.rl import make


OBS_KEYS = {"discrete_observations", "continuous_observations", "image_observations", "text_observations"}


@pytest.mark.parametrize("task_name", ["atari-alien", "babyai-action-obj-door", "metaworld-assembly", "mujoco-ant"])
def test_make(task_name):
    env = make(task_name)
    observation, info = env.reset()
    for _ in range(10):
        action = np.array(env.action_space.sample())
        observation, reward, terminated, truncated, info = env.step(action)
        assert isinstance(info, dict)
        assert set(observation.keys()).issubset(OBS_KEYS)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
