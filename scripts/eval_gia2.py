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
from gymnasium import spaces
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from gia2.configuration_gia2 import Gia2Config
from gia2.modeling_gia2 import Gia2Model
from gia2.utils import push_to_hub, save_video_grid, suppress_stdout
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
    model = Gia2Model.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir).to(device)

    video_list = []
    input_fps = []
    model_scores_dict = {}

    for task in tqdm(eval_args.tasks, desc="Evaluation", unit="task", leave=True):
        # Create the environment
        env_kwargs = {}
        if task.startswith("atari"):
            env_kwargs["clip_reward"] = False
        if eval_args.save_video:
            env_kwargs["render_mode"] = "rgb_array"
        with suppress_stdout():  # avoid printing the env info
            env = make(task, **env_kwargs)

        action_size = env.action_space.shape[0] if isinstance(env.action_space, spaces.Box) else env.action_space.n
        model_scores_dict[task] = []
        frames = []
        for episode in tqdm(range(eval_args.num_episodes), desc=task, unit="episode", leave=False):
            observation, _ = env.reset()
            observations = {key: [val] for key, val in observation.items()}
            action_key = "continuous_actions" if isinstance(env.action_space, spaces.Box) else "discrete_actions"
            actions = {action_key: []}
            ep_return = 0
            done = False
            while not done:
                action = model.get_next_action(**observations, **actions, action_size=action_size)
                observation, reward, termined, truncated, info = env.step(action)
                done = termined or truncated

                # Handle "fake done" for atari
                if done and task.startswith("atari"):
                    if "episode" not in info:
                        observation, info = env.reset()
                        done = False
                    else:
                        print("Episode done, score:", info["episode"]["r"], ep_return + reward)

                # Store the observation and action
                for key, val in observation.items():
                    observations[key].append(val)
                actions[action_key].append(action)

                # Update the return
                ep_return += reward

                # Render the environment
                if eval_args.save_video:
                    frames.append(np.array(env.render(), dtype=np.uint8))

            model_scores_dict[task].append(ep_return)
        env.close()

        # Get the mean and std of the expert and random scores
        with open("gia/eval/rl/scores_dict.json", "r") as file:
            scores_dict = json.load(file)

        expert_mean = scores_dict[task]["expert"]["mean"]
        random_mean = scores_dict[task]["random"]["mean"]

        # Normalize the scores
        raw_mean = np.mean(model_scores_dict[task])
        raw_std = np.std(model_scores_dict[task])
        norm_mean = (raw_mean - random_mean) / (expert_mean - random_mean)
        norm_std = raw_std / (expert_mean - random_mean)

        # Print the results
        tqdm.write(
            f"Task {task} Raw score: {raw_mean:.2f} ± {raw_std:.2f} "
            f"Normalized score: {norm_mean:.2f} ± {norm_std:.2f}"
        )

        # Save the video
        if eval_args.save_video:
            video_list.append(frames)
            input_fps.append(env.metadata["render_fps"])

    # Save the video
    if eval_args.save_video:
        save_video_grid(
            video_list, input_fps, f"{model_args.model_name_or_path}/replay.mp4", output_fps=30, max_length_seconds=180
        )

    # Push the model to the hub
    if eval_args.push_to_hub:
        assert eval_args.repo_id is not None, "You need to specify a repo_id to push to."
        Gia2Model.register_for_auto_class("AutoModelForCausalLM")
        Gia2Config.register_for_auto_class()
        # As long as the the trainer does not use tokenizer, we mannually save it
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer.push_to_hub(eval_args.repo_id)
        push_to_hub(model_args.model_name_or_path, eval_args.repo_id, scores_dict=model_scores_dict)
        print(f"Pushed model to https://huggingface.co/{eval_args.repo_id}")


if __name__ == "__main__":
    main()
