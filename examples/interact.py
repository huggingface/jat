import gym
import numpy as np
import torch

from gia.config import Arguments
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor


def run():
    num_envs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Arguments(output_dir="./", task_names=["mujoco-ant"])
    processor = GiaProcessor(args)
    model = GiaModel(args).to(device)
    env = gym.vector.make("Ant-v4", num_envs)
    action_dim = env.action_space.shape[1]
    obs_dim = env.observation_space.shape[1]

    # Buffer
    observations = np.empty((num_envs, 0, obs_dim))
    actions = np.empty((num_envs, 0, action_dim))

    # Prompt buffer
    # TODO: Prompt the buffer

    observation, info = env.reset()

    for i in range(100):
        observations = np.concatenate([observations, observation[:, None, :]], axis=1)

        # Compute the output of the model
        processed = processor(continuous_observations=observations, continuous_actions=actions, padding=False)
        # To torch tensors
        for key in processed.keys():
            processed[key] = torch.as_tensor(processed[key], device=device)
        action = model.generate(**processed, num_tokens=action_dim)
        # action_tokens = output.logits[:, -action_dim:].argmax(2).cpu().numpy()
        # action = processor.tokenizer.decode_continuous(action_tokens)

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Replace the fake action with the real one
        actions = np.concatenate([actions, action[:, None, :]], axis=1)

        if terminated.any() or truncated.any():
            break

    env.close()


if __name__ == "__main__":
    run()
