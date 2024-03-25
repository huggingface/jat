#!/usr/bin/env python3
"""Train a JAT model on the JAT dataset"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets.config
from datasets import load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from jat.eval.rl.core import TASK_NAME_TO_ENV_ID
from jat.modeling_jat import JatModel
from jat.utils import mix_iterable_datasets


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 10000


logger = logging.getLogger(__name__)


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
SAMPLE_WEIGHTS = {
    "conceptual-captions": 10.0,
    "oscar": 10.0,
    "wikipedia": 10.0,
}

os.environ["WANDB_ENTITY"] = "jat-project"
os.environ["WANDB_PROJECT"] = "jat"


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = JatModel(config)
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
    # Automatic cache is broken for parquet datasets
    # The following is a fix from https://github.com/huggingface/datasets/issues/3547#issuecomment-1252503988
    dataset_dict = {}
    if HF_DATASETS_OFFLINE:
        for task in tasks:
            if not os.path.exists(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset/{task}"):
                raise ValueError(
                    f"""Dataset {task} not found in {HF_DATASETS_CACHE}/jat-project/jat-dataset/
Make sure to download and save it first with
```
from datasets import load_dataset
dataset = load_dataset('jat-project/jat-dataset', '{task}')
dataset.save_to_disk('{HF_DATASETS_CACHE}/jat-project/jat-dataset/{task}')
```"""
                )
            dataset = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset/{task}")
            dataset_dict[task] = {s: d.to_iterable_dataset() for s, d in dataset.items()}
    else:
        for task in tasks:
            dataset_dict[task] = load_dataset("jat-project/jat-dataset", task, streaming=True)

    # Preprocess the dataset
    for task in dataset_dict.keys():
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

    train_dataset = {t: d["train"] for t, d in dataset_dict.items()}
    eval_dataset = {t: d["test"] for t, d in dataset_dict.items()}

    for key in tasks:  # Reduce the number of eval samples
        eval_dataset[key] = eval_dataset[key].take(data_args.eval_num_samples)

    weights = [SAMPLE_WEIGHTS.get(t, 1.0) for t in train_dataset.keys()]
    train_dataset = mix_iterable_datasets(
        list(train_dataset.values()), batch_size=training_args.per_device_train_batch_size, weights=weights
    )
    # Due to the train dataset's structure, where every 'n' consecutive samples share the same modalities, we can't
    # load all samples at once. Different sets of 'n' samples have different modalities. Therefore, we must load and
    # process each set of 'n' samples separately.
    if training_args.dispatch_batches is not False:
        raise ValueError("Make sure to pass `--dispatch_batches False`.")

    # Why the training continue after exauhsting the dataset? https://github.com/huggingface/transformers/issues/26635
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=processor
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
