from dataclasses import dataclass
from typing import Any, Optional, List, Dict

import torch
from datasets import load_dataset
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers import FeatureExtractionMixin, GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel, ProcessorMixin
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.modeling_outputs import ModelOutput

# Processor: everything related to padding, truncation, resize, normalization, etc.
# Embedding: takes the output of the processor and embeds it into a vector, must be multimodal, the input must still be episodes (conitnuous_observations, discrete_observations, text_observations, image_observations, continuous_actions, discrete_actions, rewards)

# class ContinuousEmbedding(nn.Module):
#     def __init__(self, continuous_max_size, discrete_max_val, hidden_size):


# class _MyProcessor(FeatureExtractionMixin):
#     def __init__(self, max_length):
#         self.max_length = max_length

#     def __call__(self, batch):
#         # Truncated to the max length
#         for k, v in batch.items():
#             batch[k] = [x[: self.max_length] for x in v]

#         # Pad to the max length
#         for k, v in batch.items():
#             batch[k] = [x + [x[0]] * (self.max_length - len(x)) for x in v]
#         batch["attention_mask"] = [[1] * len(x) + [0] * (self.max_length - len(x)) for x in v]
#         return batch


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

        # Encoders
        self.continuous_encoder = nn.Linear(config.continuous_max_size, config.hidden_size)
        self.discrete_embedding = nn.Embedding(config.discrete_max_val, config.hidden_size)

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.continuous_decoder = nn.Linear(config.hidden_size, config.continuous_max_size, bias=False)
        self.discrete_decoder = nn.Linear(config.hidden_size, config.discrete_max_val, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def interleave_tensors(self, tensor_dict):
        # Concatenate tensors along the T_i dimension
        concatenated_tensor = torch.cat(list(tensor_dict.values()), dim=2)

        # Flatten the concatenated tensor
        interleaved_tensor = torch.flatten(concatenated_tensor, start_dim=1, end_dim=2)

        return interleaved_tensor

    def deinterleave_tensors(self, interleaved_tensor, tensor_dict):
        # Calculate split sizes
        split_sizes = [t.shape[2] for t in tensor_dict.values()]
        # Repeat the split sizes to account for the interleaved format
        repeated_split_sizes = split_sizes * next(iter(tensor_dict.values())).shape[1]
        repeated_split_sizes[0] -= 1

        # Split the tensor
        splits = torch.split(interleaved_tensor, repeated_split_sizes, dim=1)

        # Re-assemble each tensor
        reconstructed_dict = {}
        keys = list(tensor_dict.keys())
        for i, key in enumerate(keys):
            reconstructed_dict[key] = torch.stack(splits[i :: len(tensor_dict)], dim=1)

        return reconstructed_dict

    def forward(
        self,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        text_observations: Optional[FloatTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        return_loss=True,
    ):
        inputs_embeds_dict = {}
        if continuous_observations is not None:
            # Pad observations with zeros
            batch_size, _seq_len, observation_size = continuous_observations.shape
            padded_observations = nn.functional.pad(
                continuous_observations, (0, self.config.continuous_max_size - observation_size)
            )
            encoded_continuous_observations = self.continuous_encoder(padded_observations).unsqueeze(2)
            inputs_embeds_dict["continuous_observations"] = encoded_continuous_observations

        if discrete_observations is not None:
            batch_size, _seq_len = discrete_observations.shape
            encoded_discrete_observations = self.discrete_embedding(padded_observations).unsqueeze(2)
            inputs_embeds_dict["discrete_observations"] = encoded_discrete_observations

        if continuous_actions is not None:
            # Pad actions and actions with zeros and mask them
            batch_size, _seq_len, action_size = continuous_actions.shape
            padded_actions = nn.functional.pad(continuous_actions, (0, self.config.continuous_max_size - action_size))
            encoded_continuous_actions = self.continuous_encoder(padded_actions).unsqueeze(2)
            inputs_embeds_dict["continuous_actions"] = encoded_continuous_actions

        if discrete_actions is not None:
            batch_size, _seq_len = discrete_actions.shape
            encoded_discrete_actions = self.discrete_embedding(padded_actions).unsqueeze(2)
            inputs_embeds_dict["discrete_actions"] = encoded_discrete_actions

        # Interleave observations and actions
        inputs_embeds = self.interleave_tensors(inputs_embeds_dict)

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]
        # Shift so that tokens < n predict n
        hidden_states = hidden_states[:, 1:].contiguous()
        outputs = self.deinterleave_tensors(hidden_states, inputs_embeds_dict)

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
    continuous_max_size = 6
    num_timesteps = 3

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
    )
    config.continuous_max_size = continuous_max_size
    config.discrete_max_val = 10

    model = MyModel(config)

    # 1 sequence, 10 timesteps
    observations = torch.randn(1, num_timesteps, 4)
    actions = torch.randn(1, num_timesteps, 2)

    outputs = model(continuous_observations=observations, continuous_actions=actions)

    print(observations)
    print(actions)
    print(outputs)

    # Load the dataset
    train_dataset = load_dataset("gia-project/gia-dataset", "mujoco-pendulum", split="train")
    eval_dataset = load_dataset("gia-project/gia-dataset", "mujoco-pendulum", split="test[:20]")

    # train_dataset = train_dataset.map(MyProcessor(config.window_size), batched=True, batch_size=3)
    # eval_dataset = eval_dataset.map(MyProcessor(config.window_size), batched=True, batch_size=3)

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

    trainer = Trainer(model=model.to("cuda"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()
