from dataclasses import dataclass, field
from typing import List, Optional, Union


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
    model_name: str = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "The name of the model"})
    vocab_size: Optional[int] = field(
        default=32_000 + 1024 + 1,
        metadata={"help": "The size of the model vocabulary, including the tokenization of observations"},
    )
    seq_len: int = field(default=1024, metadata={"help": "The length (number of tokens) of a sequence."})
    max_position_embeddings: Optional[int] = field(
        default=32, metadata={"help": "The maximum number of position embeddings."}
    )
    text_vocab_size: int = field(default=32_000, metadata={"help": "The size of the model vocabulary for text."})
    nb_bins: int = field(
        default=1024, metadata={"help": "The number of bins for the discretization of continuous observations."}
    )
    max_nb_observation_tokens: int = field(
        default=512, metadata={"help": "The maximum number of tokens for one observation."}
    )
    use_separator: bool = field(
        default=True, metadata={"help": "Whether to include a separator token between observations and actions."}
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
    use_pretrained: bool = field(default=True, metadata={"help": "Whether to use a pretrained model or not."})
    embed_dim: Union[int, str] = field(
        default="auto", metadata={"help": "The embedding dimension. If auto, it is set to the model size."}
    )


@dataclass
class DatasetArguments:
    task_name: str = field(default="mujoco-ant", metadata={"help": "The name of the task to load."})
    seq_len: int = field(default=1024, metadata={"help": "The length (number of tokens) of a sequence."})
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
    token_shift: int = field(
        default=32_000, metadata={"help": "The token shift for continuous and discrete observations."}
    )
    use_separator: bool = field(
        default=True, metadata={"help": "Whether to include a separator token between observations and actions."}
    )
    load_from_cache: Optional[bool] = field(
        default=True, metadata={"help": "Whether to load the dataset from the cache files."}
    )


@dataclass
class Arguments(TrainingArguments, ModelArguments, DatasetArguments):
    pass
