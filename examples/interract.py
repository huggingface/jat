import gym
import torch

from gia.config import Arguments
from gia.datasets.batch_generator import load_prompt_dataset
from gia.model.gia_model import GiaModel
from gia.processor import MultimodalProcessor
from gia.datasets.dataset_dict import DatasetDict
import numpy as np

num_envs = 2
int_per_seq = 20  # number of interactions per sequence. Hard-coded for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GiaModel(Arguments()).to(device)
env = gym.vector.make("Ant-v4", num_envs)

# For mujoco, there is one token per component of the observation and action
num_obs_tokens = env.observation_space.shape[1]
num_act_tokens = env.action_space.shape[1]

buffer = {
    "continuous_observations": torch.zeros((num_envs, int_per_seq, num_obs_tokens), dtype=torch.long, device=device),
    "continuous_observations_attention_mask": torch.zeros(
        (num_envs, int_per_seq, num_obs_tokens), dtype=torch.long, device=device
    ),
    "continuous_actions": torch.zeros((num_envs, int_per_seq, num_act_tokens), dtype=torch.long, device=device),
    "continuous_actions_attention_mask": torch.zeros(
        (num_envs, int_per_seq, num_act_tokens), dtype=torch.long, device=device
    ),
}

prompt_dataset = load_prompt_dataset("mujoco-ant")
sampled_prompts_idxs = np.random.randint(0, len(prompt_dataset), size=num_envs)

# Fill (right side) the buffer with the prompts. Truncate if necessary.
for key in buffer.keys():
    l = min(buffer[key].shape[1], prompt_dataset[key][sampled_prompts_idxs].shape[1])
    buffer[key][:, -l:] = torch.from_numpy(prompt_dataset[key][sampled_prompts_idxs, -l:]).to(device)

processor = MultimodalProcessor()

obs, info = env.reset()
for i in range(100):
    # First, roll the buffer
    for key in buffer.keys():
        buffer[key][:, :-1] = buffer[key][:, 1:]

    # Then, add the last observation to the buffer and mask the last action
    obs_tokens = processor({"continuous_observations": obs})["continuous_observations"]
    buffer["continuous_observations"][:, -1] = torch.from_numpy(obs_tokens).to(device)
    buffer["continuous_actions_attention_mask"][:, -1] = 0

    # Compute the output of the model
    output = model(buffer, eval=True)
    # TODO: use the output to sample an action
    # action = ...
    action = env.action_space.sample()

    # Add the action to the buffer and unmask it
    act_tokens = processor({"continuous_actions": action})["continuous_actions"]
    buffer["continuous_actions"][:, -1] = torch.from_numpy(act_tokens).to(device)
    buffer["continuous_actions_attention_mask"][:, -1] = 1

    # Step the environment
    next_obs, reward, terminated, truncated, info = env.step(action)

    obs = next_obs

env.close()
