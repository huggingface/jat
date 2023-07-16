import numpy as np
import pytest

from gia.eval.rl import make
from gia.eval.rl.envs.core import get_task_names


@pytest.mark.parametrize("task_name", get_task_names())
def test_make(task_name: str):
    num_envs = 2
    env = make(task_name, num_envs=num_envs)
    observation, info = env.reset()
    for _ in range(10):
        action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
        action = np.array([action_space.sample() for _ in range(num_envs)])
        observation, reward, terminated, truncated, info = env.step(action)
        assert reward.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert isinstance(info, dict)
