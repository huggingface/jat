#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import concatenate_datasets, load_dataset
from transformers import HfArgumentParser, TrainingArguments

from gia2.config import Gia2Config
from gia2.modeling import GIA2Model
from gia2.sampler import MyBatchSampler
from gia2.trainer import MyTrainer
from gia2.utils import collate_fn, preprocess_function
from gia.eval.rl.envs.core import TASK_NAME_TO_ENV_ID

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
    model = GIA2Model(config)

    # Set the tasks
    tasks = data_args.tasks
    if tasks in ["atari", "babyai", "metaworld", "mujoco"]:
        tasks = [env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(tasks)]

    # Load the dataset
    dataset = {task: load_dataset("gia-project/gia-dataset-parquet", task) for task in tasks}
    dataset = {
        t: d.map(preprocess_function, batched=True, fn_kwargs={"max_len": 256}, num_proc=data_args.preprocess_num_proc)
        for t, d in dataset.items()
    }
    train_dataset = {t: d["train"] for t, d in dataset.items()}
    eval_dataset = {t: d["test"] for t, d in dataset.items()}
    sampler = MyBatchSampler(train_dataset)
    train_dataset = concatenate_datasets(list(train_dataset.values()))

    # Instanciate the trainer and train
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_sampler=sampler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
