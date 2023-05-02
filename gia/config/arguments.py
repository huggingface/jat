import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transformers import HfArgumentParser


@dataclass
class DatasetArguments:
    """
    Arguments related to the dataset.
    """

    task_names: str = field(
        default="all",
        metadata={
            "help": "Comma-separated list of tasks to load. See the available tasks in "
            "https://huggingface.co/datasets/gia-project/gia-dataset. If 'all', load all the tasks. Defaults to 'all'."
        },
    )
    batch_size: int = field(default=8, metadata={"help": "The batch size."})
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the dataset. Defaults to True."})
    seq_len: int = field(default=1024, metadata={"help": "The length (number of tokens) of a sequence."})
    use_separator: bool = field(
        default=True, metadata={"help": "Whether to include a separator token between observations and actions."}
    )
    p_prompt: float = field(
        default=0.25, metadata={"help": "The probability of including a prompt at the beginning of a sequence."}
    )
    p_end: float = field(
        default=0.5, metadata={"help": "The probability of taking a prompt from the end of an episode."}
    )
    patch_size: int = field(
        default=16, metadata={"help": "The size of the patches to extract from image observations."}
    )
    mu: float = field(
        default=100, metadata={"help": "The μ parameter for the μ-law companding of continuous observations."}
    )
    M: float = field(
        default=256, metadata={"help": "The M parameter for the μ-law companding of continuous observations."}
    )
    nb_bins: int = field(
        default=1024, metadata={"help": "The number of bins for the discretization of continuous observations."}
    )
    load_from_cache: Optional[bool] = field(
        default=True, metadata={"help": "Whether to load the dataset from the cache files."}
    )


@dataclass
class ModelArguments:
    """
    Arguments related to the model.
    """

    model_name: str = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "The name of the model"})
    use_pretrained: bool = field(default=False, metadata={"help": "Whether to use a pretrained model or not."})
    embed_dim: int = field(default=768, metadata={"help": "The embedding dimension."})
    seq_len: int = field(default=1024, metadata={"help": "The length (number of tokens) of a sequence."})
    use_separator: bool = field(
        default=True, metadata={"help": "Whether to include a separator token between observations and actions."}
    )
    max_nb_observation_tokens: int = field(
        default=512, metadata={"help": "The maximum number of tokens for one observation."}
    )
    text_vocab_size: int = field(default=32_000, metadata={"help": "The size of the model vocabulary for text."})
    nb_bins: int = field(
        default=1024, metadata={"help": "The number of bins for the discretization of continuous observations."}
    )
    patch_size: int = field(
        default=16, metadata={"help": "The size of the patches to extract from image observations."}
    )
    image_vocab_size: int = field(
        default=128,
        metadata={
            "help": "The size of the model vocabulary for images. The maximum size for "
            "an image is therefore patch_size*image_vocab_size."
        },
    )
    num_res_channels: int = field(
        default=256, metadata={"help": "The number of residual channels in the image patch encoder."}
    )
    num_groups: int = field(
        default=32, metadata={"help": "The number of groups for the group normalization in the image patch encoder."}
    )


@dataclass
class TrainingArguments:
    """
    Arguments related to training and evaluation.
    """

    base_dir = "./runs/"

    model_ckpt: str = field(default="", metadata={"help": "Model name or path of model to be trained."})
    save_dir: str = field(
        default="",
        metadata={
            "help": "The directory where the model predictions and checkpoints will be written. If not set, it will "
            "be set to ./runs/run_{highest_run_index + 1}."
        },
    )
    max_train_steps: int = field(default=50_000, metadata={"help": "Maximum number of training steps."})
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate fo training."})
    weight_decay: float = field(default=0.1, metadata={"help": "Value of weight decay."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type."})
    num_warmup_steps: int = field(
        default=750,
        metadata={
            "help": "The number of warmup steps to do. This is not required by all schedulers (hence the argument "
            "being  optional), the function will raise an error if it's unset and the scheduler type requires it."
        },
    )
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "Number of gradient accumulation steps."})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    seed: int = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: int = field(
        default=1024,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )
    resume_from_checkpoint: str = field(  # FIXME:
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )

    @classmethod
    def _generate_save_dir(cls) -> str:
        # If it doesn't exist, use idx = 0
        if not os.path.exists(cls.base_dir):
            idx = 0
        # Otherwise, find the highest index and use that
        else:
            existing_run_dirs = [d.name for d in Path(cls.base_dir).iterdir() if d.is_dir()]
            run_indices = [
                int(match.group(1)) for run_dir in existing_run_dirs if (match := re.match(r"run_(\d+)", run_dir))
            ]
            idx = max(run_indices, default=0) + 1
        return f"{cls.base_dir}run_{idx}"

    def __post_init__(self):
        if self.save_dir == "":
            self.save_dir = self._generate_save_dir()
        if isinstance(self.task_names, str):
            self.task_names = self.task_names.split(",")


@dataclass
class EvalArguments:
    n_episodes: int = field(default=10, metadata={"help": "The number of eval episodes to perform"})


@dataclass
class Arguments(DatasetArguments, ModelArguments, TrainingArguments, EvalArguments):
    def save(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        out_path = Path(self.save_dir) / "args.json"
        with open(out_path, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=2)

    @classmethod
    def load(cls, load_dir: str) -> "Arguments":
        in_path = Path(load_dir) / "args.json"
        with open(in_path, "r") as infile:
            loaded_args = json.load(infile)
        return cls(**loaded_args)


def parse_args() -> Arguments:
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a YAML file,
        # let's parse it to get our arguments.
        args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))[0]  # [0] because we only have one group of args
    else:
        args = parser.parse_args_into_dataclasses()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    Arguments.save(args)
