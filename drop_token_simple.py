import json
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from torch import FloatTensor, Tensor, nn
from torch.utils.data import BatchSampler, DataLoader
from transformers import GPTNeoConfig, GPTNeoModel, GPTNeoPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from gia.eval.rl import make


LOSS_WEIGHTS = {
    "mujoco-ant": 0.00265386 / 0.01879,
    "mujoco-doublependulum": 0.00265386 / 0.003574,
    "mujoco-halfcheetah": 0.00265386 / 0.02215,
    "mujoco-hopper": 0.00265386 / 0.05969,
    "mujoco-pendulum": 0.00265386 / 0.0004398,
    "mujoco-reacher": 0.00265386 / 0.001816,
    "mujoco-swimmer": 0.00265386 / 0.03623,
    "mujoco-walker": 0.00265386 / 0.3287,
    "mujoco-pusher": 0.00265386 / 0.007077,
}


@dataclass
class MyOutput(ModelOutput):
    """ """

    pred_observations: torch.FloatTensor = None
    pred_actions: torch.FloatTensor = None
    observation_loss: Optional[Tensor] = None
    action_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class MuJoCoModel(GPTNeoPreTrainedModel):
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
        loss_weights: Optional[FloatTensor] = None,
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
            loss_fct = nn.MSELoss(reduction="none")
            observation_loss = torch.mean(loss_fct(pred_observations[:, 1:], continuous_observations[:, :-1]), (1, 2))
            observation_loss = torch.mean(observation_loss * loss_weights)
            action_loss = torch.mean(loss_fct(pred_actions, continuous_actions), (1, 2))
            action_loss = torch.mean(action_loss * loss_weights)

            return MyOutput(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=0.0 * observation_loss + 1.0 * action_loss,
            )
        else:
            return MyOutput(pred_observations=pred_observations, pred_actions=pred_actions)


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.res1 = ResBlock(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.res2 = ResBlock(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.res3 = ResBlock(128)
        self.fc = nn.Linear(128 * 10 * 10, hidden_size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.res1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.res2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.res3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 128 * 10 * 10)
        self.convt1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.res1 = ResBlock(64)
        self.convt2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.res2 = ResBlock(32)
        self.convt3 = nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 10, 10)
        x = F.relu(self.convt1(x))
        x = self.res1(x)
        x = F.relu(self.convt2(x))
        x = self.res2(x)
        x = self.convt3(x)
        return x


class AtariModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        class DualBatchReshapeWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                n1, n2 = x.shape[:2]
                x = x.view(n1 * n2, *x.shape[2:])
                x = self.module(x)
                x = x.view(n1, n2, *x.shape[1:])
                return x

        # Encoders
        self.encoder = DualBatchReshapeWrapper(ImageEncoder(self.config.hidden_size))
        self.embedding = nn.Embedding(18, self.config.hidden_size)

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.decoder = DualBatchReshapeWrapper(ImageDecoder(self.config.hidden_size))
        self.logits_decoder = nn.Linear(self.config.hidden_size, 18)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image_observations: Optional[FloatTensor] = None,
        discrete_actions: Optional[FloatTensor] = None,
        rewards: Optional[FloatTensor] = None,
        return_loss: bool = True,
        loss_weights: Optional[FloatTensor] = None,
    ):
        # to channel first and normalize
        processed_image_observations = image_observations.transpose(4, 2) / 128 - 1.0
        inputs_embeds_observations = self.encoder(processed_image_observations)
        inputs_embeds_actions = self.embedding(discrete_actions)
        batch_size, seq_len, _ = inputs_embeds_actions.shape

        # Interleave observations and actions
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[..., 1::2, :]
        hidden_actions = hidden_states[..., ::2, :]

        pred_observations = self.decoder(hidden_observations)
        pred_actions = self.logits_decoder(hidden_actions)

        observation_loss, action_loss = None, None
        if return_loss:
            obs_loss_fct = nn.MSELoss()
            observation_loss = obs_loss_fct(pred_observations[:, 1:], processed_image_observations[:, :-1])
            action_loss_fct = nn.CrossEntropyLoss()
            action_loss = action_loss_fct(
                torch.flatten(pred_actions, end_dim=-2), torch.flatten(discrete_actions, end_dim=-1)
            )
            print((pred_actions.argmax(-1) == discrete_actions).float().mean())
            return MyOutput(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=0.0 * observation_loss + 1.0 * action_loss,
            )
        else:
            return MyOutput(pred_observations=pred_observations, pred_actions=pred_actions)


class MyBatchSampler(BatchSampler):
    def __init__(self, sizes, batch_size, loss_weights=None):
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


def train_mujoco(tasks, experience):
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

    model = MuJoCoModel(config)

    # Load the dataset
    features = Features(
        {
            "continuous_observations": Sequence(Sequence(Value("float32"))),
            "continuous_actions": Sequence(Sequence(Value("float32"))),
            "rewards": Sequence(Value("float32")),
        }
    )
    all_datasets = {t: load_dataset("gia-project/gia-dataset", t, features=features) for t in tasks}
    all_datasets = {t: d.map(lambda x: {"loss_weights": LOSS_WEIGHTS[t]}) for t, d in all_datasets.items()}

    train_datasets = [dataset["train"] for dataset in all_datasets.values()]
    train_dataset = concatenate_datasets([d["train"] for d in all_datasets.values()])
    train_dataset.sizes = [len(d) for d in train_datasets]

    eval_dataset = {t: dataset["test"] for t, dataset in all_datasets.items()}
    eval_dataset = {t: d.select(range(100)) for t, d in eval_dataset.items()}  # only the first 100 samples

    args = TrainingArguments(
        experience,
        per_device_train_batch_size=1,
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


def eval_mujoco(task, experience, checkpoint):
    model = MuJoCoModel.from_pretrained(f"{experience}/checkpoint-{checkpoint}").to("cuda")
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


def train_atari(tasks, experience):
    num_layers = 8

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
        max_position_embeddings=512,
    )

    model = AtariModel(config)

    def preprocess_function(examples):
        # truncate (but reuse the truncated part to add a new example)
        max_len = 512 // 2
        out_dict = {key: [] for key in examples.keys()}
        ni = next(iter(examples.values()))
        for ep in range(len(ni)):
            for t in range(0, len(ni[ep]), max_len):
                for key in examples.keys():
                    out_dict[key].append(examples[key][ep][t : t + max_len])
        return out_dict

    # Load the dataset
    all_datasets = {t: load_dataset("gia-project/gia-dataset-parquet", t) for t in tasks}
    all_datasets = {
        t: d.map(preprocess_function, batched=True, num_proc=16, batch_size=10) for t, d in all_datasets.items()
    }
    all_datasets = {t: d.with_format(type="torch") for t, d in all_datasets.items()}
    train_dataset = concatenate_datasets([d["train"] for d in all_datasets.values()])  # type: Dataset
    eval_dataset = {t: d["test"] for t, d in all_datasets.items()}
    # eval_dataset = {t: d.select(range(100)) for t, d in eval_dataset.items()}  # only the first 100 samples

    args = TrainingArguments(
        experience,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=0.05,
        eval_delay=0,
        save_strategy="steps",
        save_steps=0.05,
        logging_steps=1_000,
        logging_first_step=True,
        num_train_epochs=3,
    )

    trainer = Trainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()


def eval_atari(task, experience, checkpoint):
    model = AtariModel.from_pretrained(f"{experience}/checkpoint-{checkpoint}").to("cuda")
    env = make(task, render_mode="rgb_array")
    frames = []
    all_returns = []

    for episode in range(1):
        observation, _ = env.reset()
        observations = [observation["image_observations"]]
        actions = []
        ep_return = 0
        action_placeholder = np.zeros(env.action_space.shape, dtype=np.int64)
        done = False
        while not done:
            with torch.inference_mode():
                image_observations = np.array(observations, dtype=np.uint8)[None, -256:]
                image_observations = torch.from_numpy(image_observations).to("cuda")
                discrete_actions = np.array([*actions, action_placeholder], dtype=np.int64)[None, -256:]
                discrete_actions = torch.from_numpy(discrete_actions).to("cuda")
                output = model(image_observations, discrete_actions, return_loss=False)
                logits = output.pred_actions[0, -1, : env.action_space.n]
                action = torch.multinomial(F.softmax(logits, dim=0), 1).item()
            observation, reward, termined, truncated, _ = env.step(action)
            frames.append(np.array(env.render(), dtype=np.uint8))
            done = termined or truncated
            observations.append(observation["image_observations"])
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
    out = cv2.VideoWriter(f"{experience}/{checkpoint}-{task}.mp4", fourcc, env.metadata["render_fps"], (160, 210))

    # Write frames to video
    for frame in frames:
        out.write(frame[..., [2, 1, 0]])

    # Release video writer
    out.release()


if __name__ == "__main__":
    mujoco = [
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
    atari = [
        # "atari-alien",
        # "atari-amidar",
        # "atari-assault",
        # "atari-asterix",
        # "atari-asteroids",
        # "atari-atlantis",
        # "atari-bankheist",
        # "atari-battlezone",
        # "atari-beamrider",
        # "atari-berzerk",
        # "atari-bowling",
        # "atari-boxing",
        "atari-breakout",
        # "atari-centipede",
        # "atari-choppercommand",
        # "atari-crazyclimber",
        # "atari-defender",
        # "atari-demonattack",
        # "atari-doubledunk",
        # "atari-enduro",
        # "atari-fishingderby",
        "atari-freeway",
        "atari-frostbite",
        # "atari-gopher",
        # "atari-gravitar",
        # "atari-hero",
        # "atari-icehockey",
        # "atari-jamesbond",
        # "atari-kangaroo",
        # "atari-krull",
        # "atari-kungfumaster",
        # "atari-montezumarevenge",
        "atari-mspacman",
        # "atari-namethisgame",
        # "atari-phoenix",
        # "atari-pitfall",
        "atari-pong",
        # "atari-privateeye",
        "atari-qbert",
        # "atari-riverraid",
        # "atari-roadrunner",
        # "atari-robotank",
        # "atari-seaquest",
        # "atari-skiing",
        # "atari-solaris",
        # "atari-spaceinvaders",
        # "atari-stargunner",
        # "atari-surround",
        # "atari-tennis",
        # "atari-timepilot",
        # "atari-tutankham",
        # "atari-upndown",
        # "atari-venture",
        # "atari-videopinball",
        # "atari-wizardofwor",
        # "atari-yarsrevenge",
        # "atari-zaxxon",
    ]

    train_atari(atari, "atari-6")
    # for task in atari:
    #     eval_atari(task, "atari-6", 5434)
