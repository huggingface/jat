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
        inputs_embeds_dict = {}

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

        observation_loss, action_loss = None, None
        if return_loss:
            loss_fct = nn.MSELoss()
            observation_loss = loss_fct(predicted_observations[:, 1:], continuous_observations[:, :-1])
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

    model = MyModel(config)

    # # Load the dataset
    from datasets import Sequence, Value, Features, concatenate_datasets
    features = Features(
            {
                "continuous_observations": Sequence(Sequence(Value("float32"))),
                "continuous_actions": Sequence(Sequence(Value("float32"))),
                "rewards": Sequence(Value("float32")),
            }
        )
    train_dataset = concatenate_datasets([load_dataset("gia-project/gia-dataset", task, features=features, split="train") for task in tasks])
    eval_dataset = {task: load_dataset("gia-project/gia-dataset", task, split="test[:100]") for task in tasks}

    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        "checkpoints/back_from_scratch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=500,
        eval_delay=0,
        save_steps=5000,
        logging_steps=100,
        logging_first_step=True,
        num_train_epochs=2,
    )

    trainer = Trainer(model=model.to("cuda"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()

    # Test the model
    task = "mujoco-walker"
    model = MyModel.from_pretrained("old_script_all_mujoco/checkpoint-110000").to("cuda")

    from gia.eval.rl import make
    import numpy as np

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