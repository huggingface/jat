#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets.config
import torch
import torch.profiler
from datasets import load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments, TrainerCallback

from gia.eval.rl.envs.core import TASK_NAME_TO_ENV_ID
from gia2.modeling_gia2 import Gia2Model
from gia2.utils import mix_iterable_datasets


class MyCallback(TrainerCallback):
    def __init__(self, profiler) -> None:
        super().__init__()
        self.profiler = profiler

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        self.profiler.step()


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 2000

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


LOSS_WEIGHTS = {
    "mujoco-pendulum": 20.0,
    "mujoco-doublependulum": 10.0,
}


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
    model = Gia2Model(config)
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
            if not os.path.exists(f"{HF_DATASETS_CACHE}/gia-project/gia-dataset-parquet/{task}"):
                raise ValueError(
                    f"""Dataset {task} not found in {HF_DATASETS_CACHE}/gia-project/gia-dataset-parquet/
Make sure to download and save it first with
```
from datasets import load_dataset
dataset = load_dataset('gia-project/gia-dataset-parquet', '{task}')
dataset.save_to_disk('{HF_DATASETS_CACHE}/gia-project/gia-dataset-parquet/{task}')
```"""
                )
            dataset = load_from_disk(f"{HF_DATASETS_CACHE}/gia-project/gia-dataset-parquet/{task}")
            dataset_dict[task] = {s: d.to_iterable_dataset() for s, d in dataset.items()}
    else:
        for task in tasks:
            if task == "oscar":
                dataset_dict[task] = load_dataset("ClementRomac/cleaned_deduplicated_oscar", streaming=True)
            else:
                dataset_dict[task] = load_dataset("gia-project/gia-dataset-parquet", task, streaming=True)

    # Add loss weight #TODO
    # for task in dataset.keys():
    #     for split in dataset[task].keys():
    #         loss_weight = [LOSS_WEIGHTS.get(task, 1.0)] * len(dataset[task][split])
    #         dataset[task][split] = dataset[task][split].add_column("loss_weight", loss_weight)

    dataset_dict = {
        t: {
            s: d[s].map(
                lambda example_batch: processor(**example_batch, padding="max_length", truncation="preserve"),
                batched=True,
                batch_size=10,
                remove_columns={"text", "images"}.intersection(d[s].column_names),
            )
            for s in d.keys()
        }
        for t, d in dataset_dict.items()
    }

    train_dataset = {t: d["train"] for t, d in dataset_dict.items()}
    eval_dataset = {t: d["test"] for t, d in dataset_dict.items()}

    if "oscar" in tasks:  # Reduce the number of eval samples for oscar
        eval_dataset["oscar"] = eval_dataset["oscar"].take(100)

    train_dataset = mix_iterable_datasets(train_dataset.values(), batch_size=8)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/trace"),
        record_shapes=True,
        with_stack=True,
    ) as profiler:
        # Instanciate the trainer and train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
            callbacks=[MyCallback(profiler)],
        )
        trainer.train()


if __name__ == "__main__":
    main()
