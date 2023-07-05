import numpy as np
import pytest
from gymnasium import spaces

from gia import GiaConfig, GiaModel
from gia.datasets import GiaDataCollator
from gia.eval.rl.gia_agent import GiaAgent
from gia.processing import GiaProcessor


@pytest.mark.parametrize("observation_space", [spaces.Box(low=0, high=1, shape=(4,)), spaces.MultiDiscrete([4, 5])])
@pytest.mark.parametrize("action_space", [spaces.Box(low=0, high=1, shape=(4,)), spaces.Discrete(3)])
def test_gia_agent(observation_space, action_space):
    num_envs = 2
    config = GiaConfig(seq_len=128, hidden_size=32, nul_layers=4, num_heads=4)
    model = GiaModel(config)
    processor = GiaProcessor()
    collator = GiaDataCollator()
    agent = GiaAgent(
        model=model,
        processor=processor,
        collator=collator,
        observation_space=observation_space,
        action_space=action_space,
    )
    agent.reset(num_envs)
    observations = np.array([observation_space.sample() for _ in range(num_envs)], dtype=observation_space.dtype)
    actions = agent.get_action(observations)
    for action in actions:
        assert action_space.contains(action)
