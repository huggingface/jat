#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset
from torchvision.transforms.functional import to_tensor
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from gia.eval.rl.envs.core import TASK_NAME_TO_ENV_ID
from gia2.configuration_gia2 import Gia2Config
from gia2.modeling_gia2 import Gia2Model
from gia2.utils import collate_fn, mix_iterable_datasets


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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    preprocess_num_proc: int = field(
        default=1, metadata={"help": "Number of processes to use for preprocessing the data."}
    )


LOSS_WEIGHTS = {
    "mujoco-pendulum": 20.0,
    "mujoco-doublependulum": 10.0,
}


def transforms(examples):
    # Remove keys with lists containing only None values
    examples = {k: v for k, v in examples.items() if not all(item is None for item in v)}
    if "image_observations" in examples:
        for ep_idx, episode in enumerate(examples["image_observations"]):
            examples["image_observations"][ep_idx] = [(to_tensor(img) - 0.5) / 0.5 for img in episode]
    if "image" in examples:
        examples["image"] = [(to_tensor(img) - 0.5) / 0.5 for img in examples["image"]]
    return examples


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

    config = Gia2Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = Gia2Model(config)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, trust_remote_code=True
    )

    # Set the tasks
    tasks = data_args.tasks
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    # Load the dataset
    if "oscar" in tasks:
        dataset = {"oscar": load_dataset("ClementRomac/cleaned_deduplicated_oscar", streaming=True)}
        tasks.remove("oscar")
    else:
        dataset = {}

    for _task in tasks:
        dataset[_task] = load_dataset("gia-project/gia-dataset-parquet", _task, streaming=True)

    # Add loss weight
    # for task in dataset.keys():
    #     for split in dataset[task].keys():
    #         loss_weight = [LOSS_WEIGHTS.get(task, 1.0)] * len(dataset[task][split])
    #         dataset[task][split] = dataset[task][split].add_column("loss_weight", loss_weight)

    dataset = {
        t: d.map(
            lambda example_batch: processor(**example_batch, padding="max_length", truncation=True),
            batched=True,
            batch_size=10,
            remove_columns={"text", "images"}.intersection(d["test"].column_names),
            num_proc=data_args.preprocess_num_proc,
        )
        for t, d in dataset.items()
    }

    train_dataset = {t: d["train"] for t, d in dataset.items()}
    eval_dataset = {t: d["test"] for t, d in dataset.items()}
    train_dataset = mix_iterable_datasets(train_dataset.values(), batch_size=8)
    # Instanciate the trainer and train
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
