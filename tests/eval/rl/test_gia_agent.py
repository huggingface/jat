import pytest

from gia import GiaConfig, GiaModel
from gia.eval.rl.gia_agent import GiaAgent
from gia.processing import GiaProcessor


@pytest.mark.parametrize("task_name", ["atari-alien", "babyai-action-obj-door", "metaworld-assembly", "mujoco-ant"])
def test_gia_agent(task_name):
    num_envs = 2
    config = GiaConfig(seq_len=128, hidden_size=32, nul_layers=4, num_heads=4)
    model = GiaModel(config)
    processor = GiaProcessor()
    agent = GiaAgent(model, processor, task_name, num_envs, use_prompt=False)
    agent.reset()
    observations = [agent.observation_space.sample() for _ in range(num_envs)]
    actions = agent.get_action(observations)
    for action in actions:
        assert agent.action_space.contains(action)