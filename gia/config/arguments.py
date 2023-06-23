import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import wandb
from transformers import HfArgumentParser, TrainingArguments

from gia.datasets import get_task_name_list


@dataclass
class DatasetArguments:
    """
    Arguments related to the dataset.
    """

    task_names: str = field(
        default="all",
        metadata={
            "help": (
                "Comma-separated list of tasks (or prefixes, e.g. 'mujuco') to load."
                "See the available tasks in https://huggingface.co/datasets/gia-project/gia-dataset. "
                "If 'all', load all the tasks. Defaults to 'all'."
            )
        },
    )
    use_separator: bool = field(
        default=True, metadata={"help": "Whether to include a separator token between observations and actions."}
    )
    p_prompt: float = field(
        default=0.25, metadata={"help": "The probability of including a prompt at the beginning of a sequence."}
    )
    p_end: float = field(
        default=0.5, metadata={"help": "The probability of taking a prompt from the end of an episode."}
    )
    min_prompt_len: int = field(default=10, metadata={"help": "The minimum length of a prompt (in tokens)."})
    max_prompt_len: int = field(
        default=1024, metadata={"help": "The maximum length of a prompt (in tokens)."}
    )  # Ideally, this should Optional[int], and set to to seq_len if None.
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
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    mask_loss_modalities: str = field(
        default="default",
        metadata={
            "help": (
                "The modalities to mask for the loss computation. Defaults to all modalities except text and actions."
                "Specify as a comma-separated list of modalities. For example, "
                "'continuous_observations,discrete_actions' will only mask the continuous observations and discrete "
                "actions."
            )
        },
    )
    local_positions_groups: str = field(
        default="default",
        metadata={
            "help": (
                "The groups of modalities for which to add local positions. Defaults to a single group containing all "
                "observations modalities (text, images, discrete and continuous observations)."
                "Specify as a comma-separated list of groups. For example, "
                "'text_observations,image_observations' will add local positions to only text and image observations."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to trained from.
    """

    model_name_or_path: str = field(
        default="gia-project/gia-1.27b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class EvalArguments:
    n_episodes: int = field(default=10, metadata={"help": "The number of eval episodes to perform"})
    max_eval_steps: int = field(
        default=-1,
        metadata={
            "help": "The number of test batches to evaluate. If -1 (default), the full test set will be evaluated."
        },
    )
    eval_checkpoints: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of tasks (or prefixes, e.g. 'mujuco') to load."
                "See the available tasks in https://huggingface.co/datasets/gia-project/gia-dataset. "
                "If 'all', load all the tasks. Defaults to 'all'."
            )
        },
    )
    auto_eval: bool = field(
        default=True, metadata={"help": "Whether to launch eval jobs while training on the cluster"}
    )


@dataclass
class WandBArguments:
    wandb_enabled: bool = field(
        default=True,
        metadata={"help": ("Whether to enable or disable WandB.")},
    )
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Tags to group and filter runs on Weights and Biases.")},
    )
    wandb_project: Optional[str] = field(
        default="gia",
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_entity: Optional[str] = field(
        default="huggingface",
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default="tr_00_some-descriptor",
        metadata={"help": ("Group multiple runs under this group name.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": ("Set this to a globally unique string (per project) corresponding to a single run of your script.")
        },
    )


@dataclass
class Arguments(DatasetArguments, ModelArguments, EvalArguments, WandBArguments, TrainingArguments):
    def save(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = Path(self.output_dir) / "args.json"
        with open(out_path, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=2)

    @classmethod
    def load(cls, load_dir: str) -> "Arguments":
        in_path = Path(load_dir) / "args.json"
        with open(in_path, "r") as infile:
            loaded_args = json.load(infile)
        return cls(**loaded_args)

    def __post_init__(self):
        super().__post_init__()
        # We could have the following in Dataset args and call another super post init ?
        self.task_names = get_task_name_list(self.task_names)
        if "," in self.mask_loss_modalities:
            self.mask_loss_modalities = self.mask_loss_modalities.split(",")
        if "," in self.local_positions_groups:
            self.local_positions_groups = self.local_positions_groups.split(",")
        if self.max_prompt_len is None:
            self.max_prompt_len = self.seq_len
        if self.eval_checkpoints is not None:
            if "," in self.eval_checkpoints:
                self.eval_checkpoints = self.eval_checkpoints.split(",")
            else:
                self.eval_checkpoints = [self.eval_checkpoints]

        if self.wandb_enabled:
            os.environ["WANDB_ENTITY"] = self.wandb_entity
            os.environ["WANDB_PROJECT"] = self.wandb_project
            os.environ["WANDB_RUN_GROUP"] = self.wandb_run_group
            if self.wandb_run_id is not None:
                os.environ["WANDB_RUN_ID"] = self.wandb_run_id
            if self.wandb_tags is not None:
                os.environ["WANDB_TAGS"] = ",".join(tag for tag in self.wandb_tags)

    @staticmethod
    def parse_args() -> "Arguments":
        parser = HfArgumentParser(Arguments)
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            # [0] because we only have one group of args
            args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))[0]
        else:
            args = parser.parse_args_into_dataclasses()[0]

        return args


if __name__ == "__main__":
    args = Arguments.parse_args()
    Arguments.save(args)
