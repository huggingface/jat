import pytest

from gia import GiaConfig, GiaModel
from gia.eval.rl.gia_agent import GiaAgent
from gia.processing import GiaProcessor
from gia.eval.rl import make
import gymnasium as gym
import torch

@pytest.mark.parametrize("task_name", ["atari-alien", "babyai-action-obj-door", "metaworld-assembly", "mujoco-ant"])
def test_gia_agent(task_name):
    num_envs = 2

    def env_func():
        return make(task_name)

    vec_env = gym.vector.SyncVectorEnv(env_fns=[env_func for _ in range(num_envs)])
    config = GiaConfig(seq_len=128, hidden_size=32, nul_layers=4, num_heads=4)
    model = GiaModel(config)
    processor = GiaProcessor()
    agent = GiaAgent(model, processor, task_name, num_envs, use_prompt=False)

    # Initialize the environment and the agent
    obs, info = vec_env.reset()
    agent.reset()

    # Run the agent for 100 steps
    all_episode_rewards = []
    for t in range(1_000):
        with torch.inference_mode():
            action = agent.get_action(obs)
        obs, reward, truncated, terminated, info = vec_env.step(action)
        if terminated or truncated:
            obs, info = vec_env.reset()
            agent.reset()
            print(episode_reward)
            all_episode_rewards.append(episode_reward)
            episode_reward = 0
    vec_env.close()


    observations = np.array([agent.observation_space.sample() for _ in range(num_envs)])
    actions = agent.get_action(observations)
    for action in actions:
        assert agent.action_space.contains(action)
