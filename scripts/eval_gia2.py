#!/usr/bin/env python3
"""Eval a GIA model on the GIA dataset"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import HfArgumentParser

from gia2.modeling import GIA2Model
from gia2.utils import push_to_hub, save_video_grid
from gia.eval.rl import make


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    use_cpu: bool = field(default=False, metadata={"help": "Use CPU instead of GPU."})
    save_video: bool = field(default=False, metadata={"help": "Save video of the evaluation."})
    num_episodes: int = field(default=2, metadata={"help": "Number of episodes to evaluate on."})
    push_to_hub: bool = field(default=False, metadata={"help": "Push the model to the hub."})
    repo_id: Optional[str] = field(default=None, metadata={"help": "Repository ID to push to."})


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    parser = HfArgumentParser((ModelArguments, EvaluationArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cpu") if eval_args.use_cpu else get_default_device()
    model = GIA2Model.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir).to(device)

    video_list = []
    model_scores_dict = {}

    for task in tqdm(eval_args.tasks, desc="Evaluation", unit="task", leave=True):
        env = make(task, render_mode="rgb_array" if eval_args.save_video else None)
        action_placeholder = np.zeros(env.action_space.shape, dtype=np.float32)
        all_returns = []
        frames = []
        for episode in tqdm(range(eval_args.num_episodes), desc=task, unit="episode", leave=False):
            observation, _ = env.reset()
            observations = [observation["continuous_observations"]]
            actions = []
            ep_return = 0
            done = False
            while not done:
                continuous_observations = (
                    torch.from_numpy(np.array(observations, dtype=np.float32)).unsqueeze(0).to(device)
                )
                continuous_actions = (
                    torch.from_numpy(np.array([*actions, action_placeholder], dtype=np.float32))
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.inference_mode():
                    output = model(
                        continuous_observations=continuous_observations[:, -256:],
                        continuous_actions=continuous_actions[:, -256:],
                        return_loss=False,
                    )
                    action = output.pred_actions[0, -1].cpu().numpy()
                observation, reward, termined, truncated, _ = env.step(action)
                done = termined or truncated
                observations.append(observation["continuous_observations"])
                actions.append(action)
                ep_return += reward

                if eval_args.save_video:
                    frames.append(np.array(env.render(), dtype=np.uint8))

            all_returns.append(ep_return)
        env.close()

        # Get the mean and std of the expert and random scores
        with open("gia/eval/rl/scores_dict.json", "r") as file:
            scores_dict = json.load(file)

        expert_mean = scores_dict[task]["expert"]["mean"]
        random_mean = scores_dict[task]["random"]["mean"]

        # Normalize the scores
        raw_mean = np.mean(all_returns)
        raw_std = np.std(all_returns)
        norm_mean = (raw_mean - random_mean) / (expert_mean - random_mean)
        norm_std = raw_std / (expert_mean - random_mean)

        tqdm.write(
            f"Task {task} Raw score: {raw_mean:.2f} ± {raw_std:.2f} "
            f"Normalized score: {norm_mean:.2f} ± {norm_std:.2f}"
        )
        model_scores_dict[task] = all_returns

        # Save the video
        if eval_args.save_video:
            video_list.append(frames)

    if eval_args.save_video:
        save_video_grid(video_list, f"{model_args.model_name_or_path}/replay.mp4", env.metadata["render_fps"], 180)

    if eval_args.push_to_hub:
        assert eval_args.repo_id is not None, "You need to specify a repo_id to push to."
        push_to_hub(model_args.model_name_or_path, eval_args.repo_id, scores_dict=model_scores_dict)


if __name__ == "__main__":
    main()
