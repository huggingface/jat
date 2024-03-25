#!/usr/bin/env python3
"""Train a JAT model on the JAT dataset"""


import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import datasets.config
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoProcessor, HfArgumentParser

from jat.eval.rl.core import TASK_NAME_TO_ENV_ID


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 10000


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it "
                "will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    preprocess_num_proc: int = field(
        default=1, metadata={"help": "Number of processes to use for preprocessing the data."}
    )
    eval_num_samples: int = field(default=1000, metadata={"help": "Number of samples to use for evaluation."})


LOSS_WEIGHTS = {
    **{task: 10.0 for task in TASK_NAME_TO_ENV_ID.keys() if task.startswith("mujoco")},
    **{task: 50.0 for task in TASK_NAME_TO_ENV_ID.keys() if task.startswith("metaworld")},
    "mujoco-pendulum": 50.0,
    "mujoco-doublependulum": 20.0,
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    processor = AutoProcessor.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set the tasks
    tasks = data_args.tasks
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    # Load the dataset
    dataset_dict = {}
    for task in tasks:
        dataset = load_dataset("jat-project/jat-dataset", task, streaming=True)
        if task == "oscar":
            dataset = DatasetDict({"train": dataset["train"].take(1_000_000), "test": dataset["test"].take(1_000)})

        dataset_dict[task] = dataset

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    configs = datasets.get_dataset_config_names("jat-project/jat-dataset-tokenized")

    for task in dataset_dict.keys():
        if task in configs:
            print(f"Task {task} already processed, skipping...")
            continue
        else:
            print(f"Task {task} not processed yet, processing...")
        task_dataset = {}
        for split in dataset_dict[task].keys():
            dataset = dataset_dict[task][split]
            column_names = set(dataset.column_names)  # need to be done here because this info is lost after the map
            dataset = dataset.filter(lambda example: example.get("rewards") != [])

            # Add an initial 0 reward and remove the last reward
            def add_initial_reward(example):
                if "rewards" in example:
                    example["rewards"] = [0.0] + example["rewards"][:-1]
                return example

            dataset = dataset.map(add_initial_reward)

            # We've shown that reducing the sequence length for atari doesn't impact performance but allows for a
            # larger global batch size
            max_length = 64 if task.startswith("atari") else None

            def preprocess(example_batch, max_length):
                return processor(**example_batch, padding="max_length", truncation="preserve", max_length=max_length)

            dataset = dataset.map(
                preprocess,
                batched=True,
                batch_size=1,  # small to avoid OOM
                remove_columns={"text", "images", "text_observations"}.intersection(column_names),
                fn_kwargs={"max_length": max_length},
            )

            def add_loss_weight(example, loss_weight):
                example["loss_weight"] = [loss_weight] * len(next(iter(example.values())))
                return example

            dataset = dataset.map(add_loss_weight, fn_kwargs={"loss_weight": LOSS_WEIGHTS.get(task, 1.0)})
            dataset_dict[task][split] = dataset

            print(f"Generated examples for {task}/{split}")
            task_dataset[split] = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset))
        task_dataset = DatasetDict(task_dataset)

        print(f"Pushing {task} to the hub...")
        task_dataset.push_to_hub("jat-project/jat-dataset-tokenized", config_name=task)


if __name__ == "__main__":
    main()

    # python scripts/tokenize_.py --model_name_or_path jat-project/jat-small --tasks mujoco --trust_remote_code
