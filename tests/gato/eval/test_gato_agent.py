import gymnasium as gym
import pytest
import torch

from gato import GatoConfig, GatoModel
from gato.eval.rl.gato_agent import GatoAgent
from gato.processing import GatoProcessor
from gia.eval.rl import make


@pytest.mark.parametrize("task_name", ["atari-alien", "babyai-action-obj-door", "metaworld-assembly", "mujoco-ant"])
def test_gia_agent(task_name):
    num_envs = 2

    def env_func():
        return make(task_name)

    vec_env = gym.vector.SyncVectorEnv(env_fns=[env_func for _ in range(num_envs)])
    config = GatoConfig(seq_len=128, hidden_size=32, nul_layers=4, num_heads=4)
    model = GatoModel(config)
    processor = GatoProcessor()
    agent = GatoAgent(model, processor, task_name, num_envs, use_prompt=False)

    # Initialize the environment and the agent
    obs, _ = vec_env.reset()
    agent.reset()

    # Run the agent for 1k steps
    for _ in range(1_000):
        with torch.inference_mode():
            action = agent.get_action(obs)
        obs, _, _, _, _ = vec_env.step(action)
    vec_env.close()
