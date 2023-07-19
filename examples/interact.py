import gymnasium as gym
import numpy as np
import torch
from datasets import load_dataset

from gia.config import GiaConfig
from gia.datasets import GiaDataCollator, Prompter
from gia.model.gia_model import GiaModel
from gia.processing import GiaProcessor


def run():
    num_envs = 1  # Only single env is supported for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = GiaProcessor()
    collator = GiaDataCollator()
    config = GiaConfig()
    model = GiaModel(config).to(device)
    env = gym.vector.make("Ant-v4", num_envs, render_mode="human")
    action_dim = env.action_space.shape[1]

    # Buffer intialized with a prompt
    dataset = load_dataset("gia-project/gia-dataset", "mujoco-ant", split="train[:3]")
    prompter = Prompter(dataset)
    prompts = prompter.generate_prompts(num_prompts=num_envs)
    observations = np.array(prompts["continuous_observations"])
    actions = np.array(prompts["continuous_actions"])

    observation, info = env.reset()
    done = False

    while not done:
        observations = np.concatenate([observations, observation[:, None, :]], axis=1)

        # Compute the output of the model
        processed = processor(
            continuous_observations=observations,
            continuous_actions=actions,
            padding=False,
            truncation="max_length",
            truncation_side="left",
            max_length=config.seq_len - action_dim,  # ensure not to overflow when the actions are added
        )
        # To torch tensors
        processed = collator([{key: processed[key][0] for key in processed.keys()}])  # TODO: weird syntax, to improve
        for key in processed.keys():
            processed[key] = processed[key].to(device)
        # FIXME: GPTNeo doesn't support generate with input_embeds
        action_tokens = []
        for _ in range(action_dim):  # Forward pass for each action dimension
            with torch.no_grad():
                output = model(**processed)
            logits = output["logits"]
            # Get the max logits
            last_logits = logits[:, -1, :]
            action_token = last_logits.argmax(dim=-1)
            action_tokens.append(action_token)
            # Update the input_ids
            processed["input_ids"] = torch.cat([processed["input_ids"], action_token[:, None]], dim=1)
            # Add a 1 (1, N) to (1, N+1)
            processed["input_types"] = torch.cat(
                [processed["input_types"], torch.zeros(1, 1, dtype=torch.int64, device=device)], dim=1
            )
            processed["local_positions"] = torch.cat(
                [processed["local_positions"], -torch.ones(1, 1, dtype=torch.int64, device=device)], dim=1
            )
            processed["loss_mask"] = torch.cat(
                [processed["loss_mask"], torch.ones(1, 1, dtype=torch.bool, device=device)], dim=1
            )

        action_tokens = torch.stack(action_tokens, dim=-1)

        # Decode the action tokens
        action = processor.tokenizer.decode_continuous(action_tokens.cpu().numpy())

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated.any() or truncated.any()

        # Store the action
        actions = np.concatenate([actions, np.array(action)[:, None, :]], axis=1)

    env.close()


if __name__ == "__main__":
    run()
