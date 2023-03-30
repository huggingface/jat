import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from transformers import HfArgumentParser


@dataclass
class TrainingArguments:
    """
    Configuration for training model.
    """

    model_ckpt: Optional[str] = field(default="", metadata={"help": "Model name or path of model to be trained."})
    save_dir: Optional[str] = field(
        default="./runs/run01",
        metadata={"help": "Save dir where model repo is cloned and models updates are saved to."},
    )
    dataset_name_train: Optional[str] = field(default="", metadata={"help": "Name or path of training dataset."})
    dataset_name_valid: Optional[str] = field(default="", metadata={"help": "Name or path of validation dataset."})
    train_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "Value of weight decay."})
    shuffle_buffer: Optional[int] = field(
        default=10000, metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )
    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=750, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    max_train_steps: Optional[int] = field(default=50000, metadata={"help": "Maximum number of training steps."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: Optional[int] = field(
        default=1024,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )
    tokenized: Optional[bool] = field(default=False, metadata={"help": "If True the data is pretokenized."})


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "The name of the model"})
    vocab_size: Optional[int] = field(
        default=32_000 + 1024 + 1,
        metadata={"help": "The size of the model vocabulary, including the tokenization of observations"},
    )
    max_position_embeddings: Optional[int] = field(
        default=32, metadata={"help": "The size of the model vocabulary, including the tokenization of observations"}
    )


@dataclass
class DatasetArguments:
    tasks: List[str] = field(
        default_factory=lambda: ["mujoco"], metadata={"help": "A list of tasks/envs to load in the GiaDataset"}
    )
    use_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a temporary cache to store the tokenized datasets"}
    )


@dataclass
class EvalArguments:
    n_episodes: Optional[int] = field(default=10, metadata={"help": "The number of eval episodes to perform"})


@dataclass
class Arguments(TrainingArguments, ModelArguments, DatasetArguments, EvalArguments):
    @staticmethod
    def save_args(args):
        output_dir = args.save_dir
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = Path(output_dir) / "args.json"

        with open(out_path, "w") as outfile:
            json.dump(args.__dict__, outfile, indent=2)

    @staticmethod
    def load_args(args):
        # TODO: add checking / overwriting of non-default args?
        input_dir = args.save_dir
        input_path = Path(input_dir) / "args.json"
        with open(input_path, "r") as f:
            args.__dict__ = json.load(f)

        return args


def parse_args():
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a YAML file,
        # let's parse it to get our arguments.
        args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args()

    return args
