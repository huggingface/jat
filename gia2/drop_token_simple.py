import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch import BoolTensor, FloatTensor, Tensor, nn
from transformers import GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from transformers.training_args import TrainingArguments

from gia2.data_collator import ContinuousDataCollator
from gia2.sampler import MyBatchSampler
from gia2.trainer import MyTrainer
from gia2.utils import compute_mse_loss, cyclic_expand_dim
from gia.eval.rl import make


@dataclass
class GIA2Output(ModelOutput):
    pred_observations: torch.FloatTensor = None
    pred_actions: torch.FloatTensor = None
    observation_loss: Optional[Tensor] = None
    action_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class GIA2Model(GPTNeoPreTrainedModel):
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
        attention_mask: Optional[BoolTensor] = None,
        return_loss: bool = True,
    ) -> GIA2Output:
        # Pad observations with zeros if needed
        batch_size, seq_len, obs_size = continuous_observations.shape
        continuous_observations = cyclic_expand_dim(continuous_observations, self.config.continuous_max_size)
        inputs_embeds_observations = self.continuous_encoder(continuous_observations)

        # Pad actions with zeros if needed
        batch_size, seq_len, action_size = continuous_actions.shape
        continuous_actions = cyclic_expand_dim(continuous_actions, self.config.continuous_max_size)
        inputs_embeds_actions = self.continuous_encoder(continuous_actions)

        # Interleave observations and actions repeat attention_mask accordingly
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )
        if attention_mask is not None:
            input_attention_mask = torch.repeat_interleave(attention_mask, repeats=2, dim=1)
        else:
            input_attention_mask = None

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=input_attention_mask)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[:, 1::2]
        hidden_actions = hidden_states[:, ::2]
        pred_observations = self.continuous_decoder(hidden_observations)
        pred_actions = self.continuous_decoder(hidden_actions)

        if return_loss:
            observation_loss = compute_mse_loss(
                pred_observations[:, :-1], continuous_observations[:, 1:], attention_mask[:, 1:]
            )
            action_loss = compute_mse_loss(pred_actions, continuous_actions, attention_mask)
            return GIA2Output(
                pred_observations=pred_observations[..., :obs_size],
                pred_actions=pred_actions[..., :action_size],
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=0.0 * observation_loss + 1.0 * action_loss,
            )
        else:
            return GIA2Output(
                pred_observations=pred_observations[..., :obs_size],
                pred_actions=pred_actions[..., :action_size],
            )


if __name__ == "__main__":
    from gia2.data_collator import ContinuousDataCollator

    num_layers = 8
    continuous_max_size = 27
    tasks = [
        "mujoco-ant",
        "mujoco-doublependulum",
        "mujoco-halfcheetah",
        "mujoco-hopper",
        "mujoco-pendulum",
        "mujoco-reacher",
        "mujoco-swimmer",
        "mujoco-walker",
        "mujoco-pusher",
    ]
    weights = {}

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
        max_position_embeddings=512,
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

    model = GIA2Model(config)

    # Load the dataset
    dataset = {task: load_dataset("gia-project/gia-dataset-parquet", task) for task in tasks}
    train_dataset = {task: dataset[task]["train"] for task in tasks}
    eval_dataset = {task: dataset[task]["test"] for task in tasks}
    sampler = MyBatchSampler(train_dataset, weights=weights)
    train_dataset = concatenate_datasets(list(train_dataset.values()))

    args = TrainingArguments(
        "checkpoints/v2_with_collator_all_mujoco_cyclic_fill",
        auto_find_batch_size=True,
        gradient_accumulation_steps=9,
        do_eval=True,
        eval_delay=0,
        eval_steps=0.1,
        save_steps=0.1,
        logging_steps=100,
        logging_first_step=True,
        seed=seed,
    )

    trainer = MyTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ContinuousDataCollator(continuous_max_size),
        args=args,
        train_sampler=sampler,
    )
    trainer.train()

    # Test the model
    task = "mujoco-ant"
    model = GIA2Model.from_pretrained("checkpoints/v2_with_collator_all_mujoco_cyclic_fill/checkpoint-10000").to(
        "cuda"
    )

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
                output = model(continuous_observations[:, -256:], continuous_actions[:, -256:], return_loss=False)
                action = output.pred_actions[0, -1].cpu().numpy()
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
