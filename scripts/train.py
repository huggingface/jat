import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from datasets import concatenate_datasets, load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments

from gia import GiaConfig, GiaModel
from gia.datasets import GiaDataCollator, Prompter, get_task_name_list, needs_prompt
from gia.processing import GiaProcessor


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
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

    def __post_init__(self):
        self.task_names = get_task_name_list(self.task_names)
        if "," in self.mask_loss_modalities:
            self.mask_loss_modalities = self.mask_loss_modalities.split(",")
        if "," in self.local_positions_groups:
            self.local_positions_groups = self.local_positions_groups.split(",")


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (data_args, trainer_args) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (data_args, trainer_args) = parser.parse_args_into_dataclasses()

    config = GiaConfig(
        # TODO: create model args to feed into GiaConfig, or maybe with AutoModel?
        # causal_lm_name,
        # causal_lm_config,
        # embed_dim,
        patch_size=data_args.patch_size,
        # image_vocab_size,
        # num_groups,
        # num_res_channels,
        # text_vocab_size,
        nb_bins=data_args.nb_bins,
        # max_local_position,
        use_separator=data_args.use_separator,
        # seq_len,
        # use_pretrained,
    )
    processor = GiaProcessor(
        data_args.mu,
        data_args.M,
        data_args.nb_bins,
        data_args.patch_size,
        data_args.mask_loss_modalities,
        config.seq_len,
        data_args.local_positions_groups,
        data_args.use_separator,
    )
    model = GiaModel(config)

    # Load, prompt and process the datasets
    train_datasets = {
        task_name: load_dataset("gia-project/gia-dataset", task_name, split="train")
        for task_name in data_args.task_names
    }
    prompters = {
        task_name: Prompter(dataset) for task_name, dataset in train_datasets.items() if needs_prompt(task_name)
    }

    def prompt_and_process(example, prompter: Optional[Prompter] = None):
        if prompter is not None:
            return processor(**prompter.prompt(example))
        else:
            return processor(**example)

    train_datasets = {
        task_name: dataset.map(
            partial(prompt_and_process, prompter=prompters.get(task_name)),
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=20,  # lower this from 1000 to 20 avoid OOM
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        for task_name, dataset in train_datasets.items()
    }

    train_dataset = concatenate_datasets(list(train_datasets.values()))

    # Initialize the model
    config = GiaConfig()
    model = GiaModel(config)

    # Load the dataset
    trainer = Trainer(
        model,
        trainer_args,
        data_collator=GiaDataCollator(),
        train_dataset=train_dataset,
        # eval_dataset=test_datasets,  # TODO: See https://github.com/huggingface/gia/issues/65
    )
    trainer.train()


if __name__ == "__main__":
    main()
