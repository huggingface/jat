import json
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from PIL import Image
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


def write_video(frames, filename, fps):
    """
    Writes a list of frames into a video file.

    Parameters:
    - frames (list of np.ndarray): List of frames in RGB format.
    - filename (str): Output video filename including the extension.
    - fps (int): Frames per second for the output video.
    """
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    shape = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(filename, fourcc, fps, shape)

    # Write frames to video
    for frame in frames:
        out.write(frame[..., [2, 1, 0]])  # convert RGB to BGR and write

    # Release resources
    out.release()


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
        attention_mask: Optional[FloatTensor] = None,
        return_loss: bool = True,
    ):
        # Pad observations with zeros
        batch_size, seq_len, observation_size = continuous_observations.shape
        padded_observations = F.pad(continuous_observations, (0, self.config.continuous_max_size - observation_size))
        inputs_embeds_observations = self.continuous_encoder(padded_observations)

        # Pad actions and actions with zeros and mask them
        batch_size, seq_len, action_size = continuous_actions.shape
        padded_actions = F.pad(continuous_actions, (0, self.config.continuous_max_size - action_size))
        inputs_embeds_actions = self.continuous_encoder(padded_actions)

        # Interleave observations and actions
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )
        if attention_mask is not None:
            _attention_mask = attention_mask.repeat_interleave(2, dim=1)
        else:
            _attention_mask = None

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=_attention_mask)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[..., 1::2, :]
        hidden_actions = hidden_states[..., ::2, :]

        pred_observations = self.continuous_decoder(hidden_observations)[..., :observation_size]
        pred_actions = self.continuous_decoder(hidden_actions)[..., :action_size]

        observation_loss, action_loss = None, None
        if return_loss:
            obs_criterion = nn.MSELoss(reduction="none")
            raw_loss = obs_criterion(pred_observations[:, :-1], continuous_observations[:, 1:])  # (N, L-1, K)
            raw_loss = torch.mean(raw_loss, 2)  # (N, L-1)
            masked_loss = raw_loss * attention_mask[:, 1:]  # (N, L-1)
            observation_loss = torch.sum(masked_loss) / torch.sum(attention_mask[:, 1:])

            action_criterion = nn.MSELoss(reduction="none")
            raw_loss = action_criterion(pred_actions, continuous_actions)  # (N, L, K')
            raw_loss = torch.mean(raw_loss, 2)  # (N, L)
            masked_loss = raw_loss * attention_mask  # (N, L)
            action_loss = torch.sum(masked_loss) / torch.sum(attention_mask)

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
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(num_channels)  # the batch is too self-correlated to use batch norm
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += residual
        return F.relu(out)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.res1 = ResBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.res2 = ResBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
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
        self.image_decoder = DualBatchReshapeWrapper(ImageDecoder(self.config.hidden_size))
        self.logits_decoder = nn.Linear(self.config.hidden_size, 18)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image_observations: Optional[FloatTensor] = None,
        discrete_actions: Optional[FloatTensor] = None,
        rewards: Optional[FloatTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        return_loss: bool = True,
    ):
        # to channel first and normalize
        norm_image_observations = image_observations.transpose(4, 2).transpose(3, 4) / 128 - 1.0
        inputs_embeds_observations = self.encoder(norm_image_observations)
        inputs_embeds_actions = self.embedding(discrete_actions)
        batch_size, seq_len, _ = inputs_embeds_actions.shape

        # Interleave observations and actions
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )
        if attention_mask is not None:
            _attention_mask = attention_mask.repeat_interleave(2, dim=1)
        else:
            _attention_mask = None

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=_attention_mask)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[..., 1::2, :]
        hidden_actions = hidden_states[..., ::2, :]

        pred_observations = self.image_decoder(hidden_observations)
        pred_actions = self.logits_decoder(hidden_actions)

        observation_loss, action_loss = None, None
        if return_loss:
            obs_criterion = nn.MSELoss(reduction="none")
            raw_loss = obs_criterion(pred_observations[:, :-1], norm_image_observations[:, 1:])  # (N, L-1, C, H, W)
            raw_loss = torch.mean(raw_loss, (2, 3, 4))  # (N, L-1)
            masked_loss = raw_loss * attention_mask[:, 1:]  # (N, L-1)
            observation_loss = torch.sum(masked_loss) / torch.sum(attention_mask[:, 1:])

            action_criterion = nn.CrossEntropyLoss(reduction="none")
            raw_loss = action_criterion(
                torch.flatten(pred_actions, end_dim=1), torch.flatten(discrete_actions, end_dim=1)
            )
            masked_loss = raw_loss * torch.flatten(attention_mask, end_dim=-1)
            action_loss = torch.sum(masked_loss) / torch.sum(attention_mask)

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
    continuous_max_size = 39

    config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=24,
        hidden_size=768,
        attention_types=[[["global", "local"], num_layers // 2]],
        window_size=512,
    )
    config.continuous_max_size = continuous_max_size

    model = MuJoCoModel(config)

    def preprocess_function(examples):
        max_len = 256  # 512 // 2
        out_dict = {key: [] for key in examples.keys()}
        attention_mask = []

        ni = next(iter(examples.values()))

        for ep in range(len(ni)):
            for t in range(0, len(ni[ep]), max_len):
                chunk = ni[ep][t : t + max_len]
                pad_len = max_len - len(chunk)
                for key in examples.keys():
                    pad_value = examples[key][0][0]
                    padded_chunk = examples[key][ep][t : t + max_len] + [pad_value] * pad_len
                    out_dict[key].append(padded_chunk)

                mask = [1.0] * len(chunk) + [0.0] * pad_len
                attention_mask.append(mask)
        out_dict["attention_mask"] = attention_mask
        return out_dict

    # Load the dataset
    features = Features(
        {
            "continuous_observations": Sequence(Sequence(Value("float32"))),
            "continuous_actions": Sequence(Sequence(Value("float32"))),
            "rewards": Sequence(Value("float32")),
        }
    )
    all_datasets = {t: load_dataset("gia-project/gia-dataset-parquet", t, features=features) for t in tasks}
    all_datasets = {
        t: d.map(preprocess_function, batched=True, num_proc=16, batch_size=10) for t, d in all_datasets.items()
    }
    # train_datasets = [dataset["train"] for dataset in all_datasets.values()]
    train_dataset = concatenate_datasets([d["train"] for d in all_datasets.values()])
    # train_dataset.sizes = [len(d) for d in train_datasets]

    eval_dataset = {t: dataset["test"] for t, dataset in all_datasets.items()}
    eval_dataset = {t: d.select(range(100)) for t, d in eval_dataset.items()}  # only the first 100 samples

    args = TrainingArguments(
        experience,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        auto_find_batch_size=True,
        evaluation_strategy="steps",
        eval_steps=0.05,
        eval_delay=0,
        save_strategy="steps",
        save_steps=0.05,
        logging_steps=1_000,
        logging_first_step=True,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    trainer = Trainer(model=model.to("cuda"), train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()


def eval_mujoco(task, experience, checkpoint):
    device = "cuda"
    model = MuJoCoModel.from_pretrained(f"{experience}/checkpoint-{checkpoint}").to(device)
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
                continuous_observations = torch.from_numpy(continuous_observations).to(device)
                continuous_actions = np.array([*actions, action_placeholder], dtype=np.float32)[None, ...]
                continuous_actions = torch.from_numpy(continuous_actions).to(device)
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
    print(f"Task {task} score: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"Task {task} normalized score: {mean:.2f} ± {std:.2f}")
    env.close()

    write_video(frames, f"{experience}/{checkpoint}-{task}.mp4", env.metadata["render_fps"])
    return mean


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
        max_len = 256  # 512 // 2
        out_dict = {key: [] for key in examples.keys()}
        attention_mask = []

        ni = next(iter(examples.values()))

        for ep in range(len(ni)):
            for t in range(0, len(ni[ep]), max_len):
                chunk = ni[ep][t : t + max_len]
                pad_len = max_len - len(chunk)

                for key in examples.keys():
                    if key == "discrete_actions":
                        pad_value = 0
                    elif key == "rewards":
                        pad_value = 0.0
                    elif key == "image_observations":
                        pad_value = Image.new("RGBA", (84, 84))

                    padded_chunk = examples[key][ep][t : t + max_len] + [pad_value] * pad_len
                    out_dict[key].append(padded_chunk)

                mask = [1.0] * len(chunk) + [0.0] * pad_len
                attention_mask.append(mask)
        out_dict["attention_mask"] = attention_mask
        return out_dict

    # Load the dataset
    all_datasets = {t: load_dataset("gia-project/gia-dataset-parquet", t) for t in tasks}
    all_datasets = {
        t: d.map(preprocess_function, batched=True, num_proc=16, batch_size=10, load_from_cache_file=False)
        for t, d in all_datasets.items()
    }
    all_datasets = {t: d.with_format(type="torch") for t, d in all_datasets.items()}
    train_dataset = concatenate_datasets([d["train"] for d in all_datasets.values()])  # type: Dataset
    eval_dataset = {t: d["test"] for t, d in all_datasets.items()}

    args = TrainingArguments(
        experience,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        auto_find_batch_size=True,
        evaluation_strategy="steps",
        eval_steps=0.05,
        eval_delay=0,
        save_strategy="steps",
        save_steps=0.05,
        logging_steps=1_000,
        logging_first_step=True,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )
    trainer = Trainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
    trainer.train()
    # trainer.push_to_hub()


def eval_atari(task, experience, checkpoint):
    device = "cuda"
    model = AtariModel.from_pretrained(f"{experience}/checkpoint-{checkpoint}").to(device)
    env = make(task, render_mode="human", clip_reward=False)
    frames = []
    all_returns = []
    action_placeholder = np.zeros(env.action_space.shape, dtype=np.int64)

    for episode in range(2):
        info = {}
        done = True
        observations = [None]
        actions = []
        while "episode" not in info:
            if done:
                observation, info = env.reset()
                observations[-1] = observation["image_observations"]
            image_observations = np.array(observations, dtype=np.uint8)[None, -256:]
            image_observations = torch.from_numpy(image_observations).to(device)

            # Drop the last channel to make it RGB
            tensor_rgb = image_observations[0, :, :, :, 1:]
            image_tensor = torch.cat(tuple(tensor_rgb), dim=1)

            # Convert tensor to numpy array
            tensor_np = image_tensor.cpu().numpy()

            # Create and save the PIL image
            image_pil = Image.fromarray(tensor_np, "RGB")
            image_pil.save("image_representation.png")

            discrete_actions = np.array([*actions, action_placeholder], dtype=np.int64)[None, -256:]
            discrete_actions = torch.from_numpy(discrete_actions).to(device)
            with torch.inference_mode():
                output = model(image_observations, discrete_actions, return_loss=False)
            logits = output.pred_actions[0, -1, : env.action_space.n]
            action = torch.multinomial(F.softmax(logits, dim=0), 1).item()
            observation, reward, termined, truncated, info = env.step(action)
            done = termined or truncated
            observations.append(observation["image_observations"])
            actions.append(action)
            # frames.append(np.array(env.render(), dtype=np.uint8))

        all_returns.append(info["episode"]["r"][0])

    with open("gia/eval/rl/scores_dict.json", "r") as file:
        scores_dict = json.load(file)

    expert_mean = scores_dict[task]["expert"]["mean"]
    random_mean = scores_dict[task]["random"]["mean"]

    mean = (np.mean(all_returns) - random_mean) / (expert_mean - random_mean)
    std = np.std(all_returns) / (expert_mean - random_mean)

    print(f"Task {task} score: {np.mean(all_returns)} ± {np.std(all_returns)}")
    print(f"Task {task} normalized score: {mean:.2f} ± {std:.2f}")
    env.close()

    write_video(frames, f"{experience}/{checkpoint}-{task}.mp4", env.metadata["render_fps"])

    return mean


if __name__ == "__main__":
    mujoco = [
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
        # "atari-breakout",
        # "atari-centipede",
        # "atari-choppercommand",
        # "atari-crazyclimber",
        # "atari-defender",
        # "atari-demonattack",
        # "atari-doubledunk",
        # "atari-enduro",
        # "atari-fishingderby",
        # "atari-freeway",
        # "atari-frostbite",
        # "atari-gopher",
        # "atari-gravitar",
        # "atari-hero",
        # "atari-icehockey",
        # "atari-jamesbond",
        # "atari-kangaroo",
        # "atari-krull",
        # "atari-kungfumaster",
        # "atari-montezumarevenge",
        # "atari-mspacman",
        # "atari-namethisgame",
        # "atari-phoenix",
        # "atari-pitfall",
        "atari-pong",
        # "atari-privateeye",
        # "atari-qbert",
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
    metaworld = [
        # "metaworld-assembly",
        # "metaworld-basketball",
        # "metaworld-bin-picking",
        # "metaworld-box-close",
        # "metaworld-button-press-topdown-wall",
        # "metaworld-button-press-topdown",
        # "metaworld-button-press-wall",
        # "metaworld-button-press",
        # "metaworld-coffee-button",
        # "metaworld-coffee-pull",
        # "metaworld-coffee-push",
        # "metaworld-dial-turn",
        # "metaworld-disassemble",
        "metaworld-door-close",
        # "metaworld-door-lock",
        # "metaworld-door-open",
        # "metaworld-door-unlock",
        # "metaworld-drawer-close",
        # "metaworld-drawer-open",
        # "metaworld-faucet-close",
        # "metaworld-faucet-open",
        # "metaworld-hammer",
        # "metaworld-hand-insert",
        # "metaworld-handle-press-side",
        # "metaworld-handle-press",
        # "metaworld-handle-pull-side",
        # "metaworld-handle-pull",
        # "metaworld-lever-pull",
        # "metaworld-peg-insert-side",
        # "metaworld-peg-unplug-side",
        # "metaworld-pick-out-of-hole",
        # "metaworld-pick-place-wall",
        # "metaworld-pick-place",
        # "metaworld-plate-slide-back-side",
        # "metaworld-plate-slide-back",
        # "metaworld-plate-slide-side",
        # "metaworld-plate-slide",
        # "metaworld-push-back",
        # "metaworld-push-wall",
        # "metaworld-push",
        # "metaworld-reach-wall",
        # "metaworld-reach",
        # "metaworld-shelf-place",
        # "metaworld-soccer",
        # "metaworld-stick-pull",
        # "metaworld-stick-push",
        # "metaworld-sweep-into",
        # "metaworld-sweep",
        # "metaworld-window-close",
        # "metaworld-window-open",
    ]
    train_mujoco(mujoco, "checkpoints/mujoco-fix-offest-ant-and-double-pendulum")
    # train_atari(atari, "atari-pong-transposed")
    scores = []
    for task in mujoco:
        score = eval_mujoco(task, "mujoco-fix-offest", 106_431)
        scores.append(score)
    print(f"Average score: {np.mean(scores)}")