import gym
import numpy as np

from gia.config import Arguments
from gia.datasets.batch_generator import BatchGenerator
from gia.model.gia_model import GiaModel
from gia.datasets import load_gia_dataset
from gia.processor import MultimodalProcessor

# model = GiaModel(Arguments())
env = gym.make("Ant-v4")
dataset = load_gia_dataset("mujoco-ant", p_prompt=0.0)
processor = MultimodalProcessor(mu=100, M=256, nb_bins=1024)
batch_generator = BatchGenerator()
print(dataset[0])
# prompt = batch_generator.generate_prompts(dataset, 1)
# print(prompt)
obs, info = env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    d = {
        "continuous_observations": np.array([obs]),
        "continuous_actions": np.array([action]),
        "rewards": np.array([reward]),
        "dones": np.array([terminated or truncated]),
    }
    batch_generator(d)
    if terminated or truncated:
        obs, info = env.reset()
