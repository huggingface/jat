import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from torch import FloatTensor, Tensor, nn
from transformers import GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from gia.eval.rl import make


@dataclass
class MyOutput(ModelOutput):
    predicted_observations: torch.FloatTensor = None
    predicted_actions: torch.FloatTensor = None
    observation_loss: Optional[Tensor] = None
    action_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class MyModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Encoders
        self.continuous_encoder = nn.Linear(config.continuous_max_size, config.hidden_size)

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.continuous_decoder = nn.Linear(config.hidden_size, config.continuous_max_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        continuous_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        rewards: Optional[FloatTensor] = None,
        return_loss: bool = True,
    ):
        # Pad observations with zeros
        batch_size, seq_len, observation_size = continuous_observations.shape
        padded_observations = nn.functional.pad(
            continuous_observations, (0, self.config.continuous_max_size - observation_size)
        )
        encoded_continuous_observations = self.continuous_encoder(padded_observations)
        inputs_embeds_observations = encoded_continuous_observations

        # Pad actions and actions with zeros and mask them
        batch_size, seq_len, action_size = continuous_actions.shape
        padded_actions = nn.functional.pad(continuous_actions, (0, self.config.continuous_max_size - action_size))
        encoded_continuous_actions = self.continuous_encoder(padded_actions)
        inputs_embeds_actions = encoded_continuous_actions

        # Interleave observations and actions
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[..., 1::2, :]
        hidden_actions = hidden_states[..., ::2, :]

        predicted_observations = self.continuous_decoder(hidden_observations)[..., :observation_size]
        predicted_actions = self.continuous_decoder(hidden_actions)[..., :action_size]

        if return_loss:
            loss_fct = nn.MSELoss()
            observation_loss = loss_fct(predicted_observations[:, 1:], continuous_observations[:, :-1])
            action_loss = loss_fct(predicted_actions, continuous_actions)

            return MyOutput(
                predicted_observations=predicted_observations,
                predicted_actions=predicted_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=0.0 * observation_loss + 1.0 * action_loss,
            )
        else:
            return MyOutput(predicted_observations=predicted_observations, predicted_actions=predicted_actions)


if __name__ == "__main__":
    num_layers = 8
    continuous_max_size = 27
    tasks = [
        "mujoco-ant",
        "mujoco-doublependulum",
        # "mujoco-halfcheetah",
        # "mujoco-hopper",
        # "mujoco-pendulum",
        # "mujoco-reacher",
        # "mujoco-swimmer",
        # "mujoco-walker",
        # "mujoco-pusher",
    ]

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
    )
    config.continuous_max_size = continuous_max_size

    # Set the seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = MyModel(config)

    # Load the dataset
    train_dataset = concatenate_datasets(
        [load_dataset("gia-project/gia-dataset-parquet", task, split="train") for task in tasks]
    )
    eval_dataset = {task: load_dataset("gia-project/gia-dataset-parquet", task, split="test[:100]") for task in tasks}

    args = TrainingArguments(
        "checkpoints/v2_just_action_loss",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=500,
        eval_delay=0,
        save_steps=5000,
        logging_steps=100,
        logging_first_step=True,
        num_train_epochs=2,
        seed=seed,
    )

    trainer = Trainer(model=model.to("cuda"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()

    # Test the model
    task = "mujoco-walker"
    model = MyModel.from_pretrained("old_script_all_mujoco/checkpoint-110000").to("cuda")

    env = make(task, render_mode="rgb_array")

    for episode in range(10):
        observation, _ = env.reset()
        observations = [observation["continuous_observations"]]
        actions = []
        ep_return = 0
        all_returns = []
        action_placeholder = np.zeros(env.action_space.shape, dtype=np.float32)
        done = False
        while not done:
            with torch.inference_mode():
                continuous_observations = torch.tensor(observations, dtype=torch.float32).unsqueeze(0).to("cuda")
                continuous_actions = (
                    torch.tensor([*actions, action_placeholder], dtype=torch.float32).unsqueeze(0).to("cuda")
                )
                output = model(continuous_observations, continuous_actions, return_loss=False)
                action = output.predicted_actions[0, -1].cpu().numpy()
            observation, reward, termined, truncated, _ = env.step(action)
            done = termined or truncated
            observations.append(observation["continuous_observations"])
            actions.append(action)
            ep_return += reward
        all_returns.append(ep_return)
        print(ep_return)
    score = np.array(all_returns)
    print("Score:", score.mean(), score.std())
    env.close()
