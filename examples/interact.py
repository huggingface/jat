import gym
import numpy as np
import torch
from datasets import load_dataset

from gia.config import Arguments
from gia.datasets import generate_prompts
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor


def run():
    num_envs = 1  # Only single env is supported for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Arguments(output_dir="./", task_names=["mujoco-ant"])
    processor = GiaProcessor(args)
    model = GiaModel(args).to(device)
    env = gym.vector.make("Ant-v4", num_envs)
    action_dim = env.action_space.shape[1]

    # Buffer intialized with a prompt
    dataset = load_dataset("gia-project/gia-dataset", "mujoco-ant", split="train", revision="episode_structure")
    prompts = generate_prompts(dataset, num_prompts=num_envs)
    observations = np.array(prompts["continuous_observations"])
    actions = np.array(prompts["continuous_actions"])

    observation, info = env.reset()
    done = False

    while not done:
        observations = np.concatenate([observations, observation[:, None, :]], axis=1)

        # Compute the output of the model
        processed = processor(
            continuous_observations=observations, continuous_actions=actions, padding=False, truncate="max_length"
        )
        # To torch tensors
        for key in processed.keys():
            processed[key] = torch.as_tensor(processed[key], device=device)
        # FIXME: GPTNeo doesn't support generate with input_embeds
        action_tokens = model.generate(**processed, num_tokens=action_dim)
        action = processor.tokenizer.decode_continuous(action_tokens.cpu().numpy())

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated.any() or truncated.any()

        # Store the action
        actions = np.concatenate([actions, action[:, None, :]], axis=1)

    env.close()


if __name__ == "__main__":
    run()
