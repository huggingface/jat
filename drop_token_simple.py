from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from torch import FloatTensor, LongTensor, nn
from torch.utils.data import RandomSampler
from transformers import GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_pt_utils import LengthGroupedSampler


@dataclass
class MyOutput(ModelOutput):
    """ """

    pred_discrete_observations: Optional[torch.LongTensor] = None
    pred_continuous_observations: Optional[torch.FloatTensor] = None
    pred_continuous_actions: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


class Pad1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.nn.functional.pad(x, (0, self.size - x.shape[-1]))


class OneHotFlatten(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return nn.functional.one_hot(x, num_classes=self.num_classes).flatten(-2)


class MyModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Encoders
        self.continuous_encoder = nn.Sequential(
            Pad1d(config.continuous_max_size),
            nn.Linear(config.continuous_max_size, config.hidden_size),
        )

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.continuous_decoder = nn.Linear(config.hidden_size, config.continuous_max_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        discrete_observations: Optional[LongTensor] = None,
        continuous_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        rewards: Optional[FloatTensor] = None,
        return_loss: bool = True,
    ):
        inputs_embeds = {}
        if discrete_observations is not None:
            discrete_observations_onehot = F.one_hot(discrete_observations, num_classes=self.config.discrete_max)
            discrete_observations_onehot = discrete_observations_onehot.flatten(-2).to(torch.float32)
            inputs_embeds["discrete_observations"] = self.continuous_encoder(discrete_observations_onehot)

        if continuous_observations is not None:
            inputs_embeds["continuous_observations"] = self.continuous_encoder(continuous_observations)

        if continuous_actions is not None:
            inputs_embeds["continuous_actions"] = self.continuous_encoder(continuous_actions)

        modalities = list(inputs_embeds.keys())
        num_modalities = len(modalities)

        # Interleave observations and actions
        inputs_embeds = list(inputs_embeds.values())
        inputs_embeds = torch.stack(inputs_embeds, dim=2).flatten(1, 2)

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        keys = modalities[1:] + modalities[:1]

        hidden_outputs = {}
        for idx in range(num_modalities):
            hidden_outputs[keys[idx]] = hidden_states[..., idx::num_modalities, :]

        discrete_observation_loss = None
        continuous_observation_loss = None
        continuous_action_loss = None
        pred_discrete_observations = None
        pred_continuous_observations = None
        pred_continuous_actions = None

        need_roll = True
        if discrete_observations is not None:
            logits = self.continuous_decoder(hidden_outputs["discrete_observations"])
            logits = logits[..., : discrete_observations_onehot.shape[-1]]
            logits = logits.reshape(logits.shape[0], logits.shape[1], -1, self.config.discrete_max)
            observation_probabilities = logits.softmax(dim=-1)
            pred_discrete_observations = observation_probabilities.argmax(dim=-1)
            if return_loss:
                loss_fct = nn.CrossEntropyLoss()
                if need_roll:
                    observation_probabilities = observation_probabilities[:, 1:]
                    discrete_observations = discrete_observations[:, :-1]
                    need_roll = False
                discrete_observation_loss = loss_fct(
                    torch.flatten(observation_probabilities, end_dim=-2), torch.flatten(discrete_observations)
                )

        if continuous_observations is not None:
            pred_continuous_observations = self.continuous_decoder(hidden_outputs["continuous_observations"])
            pred_continuous_observations = pred_continuous_observations[..., : continuous_observations.shape[-1]]
            if return_loss:
                loss_fct = nn.MSELoss()
                if need_roll:
                    continuous_observation_loss = loss_fct(
                        pred_continuous_observations[:, 1:], continuous_observations[:, :-1]
                    )
                    need_roll = False
                else:
                    continuous_observation_loss = loss_fct(pred_continuous_observations, continuous_observations)

        if continuous_actions is not None:
            pred_continuous_actions = self.continuous_decoder(hidden_outputs["continuous_actions"])
            pred_continuous_actions = pred_continuous_actions[..., : continuous_actions.shape[-1]]
            if return_loss:
                loss_fct = nn.MSELoss()
                continuous_action_loss = loss_fct(pred_continuous_actions, continuous_actions)

        if return_loss:
            losses = [discrete_observation_loss, continuous_observation_loss, continuous_action_loss]
            loss = sum([loss for loss in losses if loss is not None])
        else:
            loss = None

        return MyOutput(
            pred_discrete_observations=pred_discrete_observations,
            pred_continuous_observations=pred_continuous_observations,
            pred_continuous_actions=pred_continuous_actions,
            loss=loss,
        )


class MyTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_length:
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                model_input_name=self.train_dataset.column_names[0],
            )
        else:
            return RandomSampler(self.train_dataset)


def train():
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

    # Load the dataset
    features = Features(
        {
            "continuous_observations": Sequence(Sequence(Value("float32"))),
            "continuous_actions": Sequence(Sequence(Value("float32"))),
            "rewards": Sequence(Value("float32")),
        }
    )
    train_dataset = concatenate_datasets(
        [
            load_dataset(
                "gia-project/gia-dataset-parquet",
                task,
                revision="try_parquet",
                split="train",
                writer_batch_size=1,
                features=features,
            )
            for task in tasks
        ]
    )

    eval_dataset = {
        task: load_dataset(
            "gia-project/gia-dataset-parquet",
            task,
            revision="try_parquet",
            split="test[:100]",
            writer_batch_size=1,
            features=features,
        )
        for task in tasks
    }

    args = TrainingArguments(
        "test",
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        auto_find_batch_size=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        eval_delay=0,
        logging_steps=1000,
        logging_first_step=True,
        num_train_epochs=5,
        group_by_length=True,
    )

    trainer = MyTrainer(model=model.to("cpu"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()


def eval(task):
    model = MyModel.from_pretrained("test/checkpoint-81000").to("cpu")
    env = make(task, render_mode="human")

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
                continuous_observations = torch.tensor(observations, dtype=torch.float32).unsqueeze(0).to("cpu")
                continuous_actions = (
                    torch.tensor([*actions, action_placeholder], dtype=torch.float32).unsqueeze(0).to("cpu")
                )
                output = model(continuous_observations, continuous_actions, return_loss=False)
                action = output.predicted_actions[0, -1].cpu().numpy()
            observation, reward, termined, truncated, _ = env.step(action)
            done = termined or truncated
            observations.append(observation["continuous_observations"])
            actions.append(action)
            ep_return += reward
        all_returns.append(ep_return)
        # print((ep_return - 57.46) / (9338.69 - 57.46))
    score = np.array(all_returns)
    print(f"Score for {task}: {score.mean()} +- {score.std()}")
    env.close()


if __name__ == "__main__":
    # batch_size = 2
    # seq_len = 4

    # config = GPTNeoConfig()
    # config.continuous_max_size = 15
    # config.discrete_max = 3
    # model = MyModel(config)

    # discrete_obs = torch.randint(0, config.discrete_max, (batch_size, seq_len, 2))
    # continuous_obs = torch.rand(batch_size, seq_len, 5)
    # continuous_actions = torch.rand(batch_size, seq_len, 10)

    # output = model(discrete_obs, continuous_obs, continuous_actions)
    # print(output)
    train()
    # eval("mujoco-ant")
    # eval("mujoco-doublependulum")
    # eval("mujoco-halfcheetah")
    # eval("mujoco-hopper")
    # eval("mujoco-pendulum")
    # eval("mujoco-reacher")
    # eval("mujoco-swimmer")
    # eval("mujoco-walker")
    # eval("mujoco-pusher")
