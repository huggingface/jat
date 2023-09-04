import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from torch import FloatTensor, Tensor, nn
from transformers import GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from gia.eval.rl import make
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, BatchSampler, SequentialSampler
import random
from transformers.utils import is_datasets_available
import datasets
from transformers.trainer_utils import seed_worker


@dataclass
class MyOutput(ModelOutput):
    """ """

    pred_observations: torch.FloatTensor = None
    pred_actions: torch.FloatTensor = None
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

        pred_observations = self.continuous_decoder(hidden_observations)[..., :observation_size]
        pred_actions = self.continuous_decoder(hidden_actions)[..., :action_size]

        observation_loss, action_loss = None, None
        if return_loss:
            loss_fct = nn.MSELoss()
            obs_loss_weight = self.config.continuous_max_size / pred_observations.shape[-1]
            observation_loss = loss_fct(pred_observations[:, 1:], continuous_observations[:, :-1]) * obs_loss_weight
            act_loss_weight = self.config.continuous_max_size / pred_actions.shape[-1]
            action_loss = loss_fct(pred_actions, continuous_actions) * act_loss_weight

            return MyOutput(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=observation_loss + action_loss,
            )
        else:
            return MyOutput(pred_observations=pred_observations, pred_actions=pred_actions)


class MyBatchSampler(BatchSampler):
    def __init__(self, sizes, batch_size):
        self.sizes = sizes
        self.batch_size = batch_size

    def __iter__(self):
        # Create a list of indices for each dataset
        indices_list = []
        cum_sum = 0

        for size in self.sizes:
            sublist = list(range(cum_sum, cum_sum + size))
            random.shuffle(sublist)
            indices_list.append(sublist)
            cum_sum += size

        # Create a list of batches
        batches = []
        for indices in indices_list:
            # Create batches of size self.batch_size
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i : i + self.batch_size])

        # Shuffle the batches
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return sum([s // self.batch_size + (s % self.batch_size != 0) for s in self.sizes])


class MyTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            # "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["batch_sampler"] = MyBatchSampler(train_dataset.sizes, self._train_batch_size)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def train(tasks, experience):
    num_layers = 8
    continuous_max_size = 27

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
    )
    config.continuous_max_size = continuous_max_size

    model = MyModel(config)

    # Load the dataset
    features = Features(
        {
            "continuous_observations": Sequence(Sequence(Value("float32"))),
            "continuous_actions": Sequence(Sequence(Value("float32"))),
            "rewards": Sequence(Value("float32")),
        }
    )
    train_datasets = [
        load_dataset("gia-project/gia-dataset", task, features=features, split="train") for task in tasks
    ]
    train_dataset = concatenate_datasets(train_datasets)
    train_dataset.sizes = [len(dataset) for dataset in train_datasets]
    eval_dataset = {
        task: load_dataset("gia-project/gia-dataset", task, features=features, split="test[:100]") for task in tasks
    }

    args = TrainingArguments(
        experience,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=500,
        eval_delay=0,
        save_steps=5000,
        logging_steps=100,
        logging_first_step=True,
        num_train_epochs=1,
    )

    trainer = MyTrainer(model=model.to("cuda"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()


def eval(task, experience, checkpoint):
    model = MyModel.from_pretrained(f"{experience}/checkpoint-{checkpoint}").to("cuda")
    env = make(task, render_mode="rgb_array")
    frames = []
    all_returns = []

    for episode in range(2):
        observation, _ = env.reset()
        observations = [observation["continuous_observations"]]
        actions = []
        ep_return = 0
        action_placeholder = np.zeros(env.action_space.shape, dtype=np.float32)
        done = False
        while not done:
            with torch.inference_mode():
                continuous_observations = np.array(observations, dtype=np.float32)[None, ...]
                continuous_observations = torch.from_numpy(continuous_observations).to("cuda")
                continuous_actions = np.array([*actions, action_placeholder], dtype=np.float32)[None, ...]
                continuous_actions = torch.from_numpy(continuous_actions).to("cuda")
                output = model(continuous_observations, continuous_actions, return_loss=False)
                action = output.pred_actions[0, -1].cpu().numpy()
            observation, reward, termined, truncated, _ = env.step(action)
            frames.append(np.array(env.render(), dtype=np.uint8))
            done = termined or truncated
            observations.append(observation["continuous_observations"])
            actions.append(action)
            ep_return += reward
        all_returns.append(ep_return)

    with open("gia/eval/rl/scores_dict.json", "r") as file:
        scores_dict = json.load(file)

    expert_mean = scores_dict[task]["expert"]["mean"]
    random_mean = scores_dict[task]["random"]["mean"]

    mean = (np.mean(all_returns) - random_mean) / (expert_mean - random_mean)
    std = np.std(all_returns) / (expert_mean - random_mean)

    print(f"Task {task} normalized score: {mean:.2f} ± {std:.2f}")
    env.close()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{experience}/{checkpoint}-{task}.mp4", fourcc, env.metadata["render_fps"], (480, 480))

    # Write frames to video
    for frame in frames:
        out.write(frame)

    # Release video writer
    out.release()


if __name__ == "__main__":
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
    # train(tasks, "old_script_with_weights")
    for task in tasks:
        eval(task, "old_script_all_mujoco", 25_000)

