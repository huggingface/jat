from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch import Tensor, nn
from transformers import FeatureExtractionMixin, GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput


class MyProcessor(FeatureExtractionMixin):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, batch):
        # Truncated to the max length
        for k, v in batch.items():
            batch[k] = [x[: self.max_length] for x in v]

        # Pad to the max length
        for k, v in batch.items():
            batch[k] = [x + [x[0]] * (self.max_length - len(x)) for x in v]
        batch["attention_mask"] = [[1] * len(x) + [0] * (self.max_length - len(x)) for x in v]
        return batch



@dataclass
class MyOutput(ModelOutput):
    """ """

    predicted_observations: torch.FloatTensor = None
    predicted_actions: torch.FloatTensor = None
    observation_loss: Optional[Tensor] = None
    action_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class MyModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModel(config)

        # Encoder
        self.continuous_observation_encoder = nn.Linear(config.max_observation_size, config.hidden_size)
        self.continuous_action_encoder = nn.Linear(config.max_action_size, config.hidden_size)
        

        # Decoder
        self.observation_decoder = nn.Linear(config.hidden_size, config.max_observation_size)
        self.action_decoder = nn.Linear(config.hidden_size, config.max_action_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, continuous_observations, discrete_observations, text_observations, continuous_actions, return_loss=True):
        # Pad observations and actions with zeros and mask them
        batch_size, num_observations, observation_size = continuous_observations.shape
        batch_size, num_actions, action_size = continuous_actions.shape

        padded_observations = nn.functional.pad(
            continuous_observations, (0, self.config.max_observation_size - observation_size)
        )
        padded_actions = nn.functional.pad(continuous_actions, (0, self.config.max_action_size - action_size))

        encoded_observations = self.continuous_observation_encoder(padded_observations)
        encoded_actions = self.continuous_action_encoder(padded_actions)

        # interleave observations and actions
        encoded_inputs = torch.empty(
            batch_size, num_observations + num_actions, self.config.hidden_size, device=continuous_observations.device
        )
        encoded_inputs[:, 0::2] = encoded_observations
        encoded_inputs[:, 1::2] = encoded_actions

        transformer_outputs = self.transformer(inputs_embeds=encoded_inputs)

        hidden_states = transformer_outputs[0]

        # Shift so that tokens < n predict n
        hidden_observations = hidden_states[:, 1::2].contiguous()  # o2, o3, ...
        hidden_actions = hidden_states[:, 0::2].contiguous()  # a1, a2, ...

        predicted_observations = self.observation_decoder(hidden_observations)[..., :observation_size]
        predicted_actions = self.action_decoder(hidden_actions)[..., :action_size]

        observation_loss, action_loss = None, None
        if return_loss:
            loss_fct = nn.MSELoss()
            observation_loss = loss_fct(predicted_observations, continuous_observations)
            action_loss = loss_fct(predicted_actions, continuous_actions)

            return MyOutput(
                predicted_observations=predicted_observations,
                predicted_actions=predicted_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=observation_loss + action_loss,
            )
        else:
            return MyOutput(predicted_observations=predicted_observations, predicted_actions=predicted_actions)

if __name__ == "__main__":
    num_layers = 8
    max_observation_size = 6
    max_action_size = 5
    num_timesteps = 3

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
    )
    config.max_observation_size = max_observation_size
    config.max_action_size = max_action_size

    model = MyModel(config)

    # 1 sequence, 10 timesteps
    observations = torch.randn(1, num_timesteps, 4)
    actions = torch.randn(1, num_timesteps, 2)

    outputs = model(observations, actions, return_loss=True)

    print(observations)
    print(actions)
    print(outputs)

    # Load the dataset
    train_dataset = load_dataset("gia-project/gia-dataset", "mujoco-pendulum", split="train")
    eval_dataset = load_dataset("gia-project/gia-dataset", "mujoco-pendulum", split="test[:20]")



    train_dataset = train_dataset.map(MyProcessor(config.window_size), batched=True, batch_size=3)
    eval_dataset = eval_dataset.map(MyProcessor(config.window_size), batched=True, batch_size=3)


    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        "test",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=100,
        eval_delay=0,
        logging_steps=100,
        logging_first_step=True,
    )

    trainer = Trainer(model=model.to("mps"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()
