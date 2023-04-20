import gym
import numpy as np
import torch

from gia.config import Arguments
from gia.datasets import load_prompt_dataset
from gia.model.gia_model import GiaModel
from gia.processor import MultimodalProcessor


def run():
    num_envs = 1
    int_per_seq = 20  # number of interactions per sequence. Hard-coded for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Arguments()
    model = GiaModel(args).to(device)
    env = gym.vector.make("Ant-v4", num_envs)

    # For mujoco, there is one token per component of the observation and action
    num_obs_tokens = env.observation_space.shape[1]
    num_act_tokens = env.action_space.shape[1]

    buffer = {
        "continuous_observations": torch.zeros(
            (num_envs, int_per_seq, num_obs_tokens), dtype=torch.long, device=device
        ),
        "continuous_observations_attention_mask": torch.zeros(
            (num_envs, int_per_seq, num_obs_tokens), dtype=torch.long, device=device
        ),
        "continuous_actions": torch.zeros((num_envs, int_per_seq, num_act_tokens), dtype=torch.long, device=device),
        "continuous_actions_attention_mask": torch.zeros(
            (num_envs, int_per_seq, num_act_tokens), dtype=torch.long, device=device
        ),
    }

    prompt_dataset = load_prompt_dataset("mujoco-ant", args)
    sampled_prompts_idxs = np.random.randint(0, len(prompt_dataset), size=num_envs)
    

    # Fill (right side) the buffer with the prompts. Truncate if necessary.
    for key in buffer.keys():
        sampled_prompts = prompt_dataset[key][sampled_prompts_idxs]
        prompt_length = min(buffer[key].shape[1], sampled_prompts.shape[1]) # truncate if prompt is too long
        buffer[key][:, -prompt_length:] = torch.from_numpy(sampled_prompts[:, -prompt_length:]).to(device)

    processor = MultimodalProcessor(args)

    obs, info = env.reset()
    for i in range(100):
        # First, roll the buffer
        for key in buffer.keys():
            buffer[key][:, :-1] = buffer[key][:, 1:].clone()

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


if __name__ == "__main__":
    run()
