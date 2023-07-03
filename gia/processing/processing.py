from typing import Dict, List, Optional, Tuple, TypeVar, Union

import datasets
import numpy as np
from PIL import Image

from .interleaver import Interleaver
from .local_positions_adder import LocalPositionsAdder
from .tokenizer import GiaTokenizer
from .utils import nested_like


T = TypeVar("T")
NestedList = Union[None, T, List["NestedList[T]"]]


class GiaProcessor:
    """
    Processor for GIA

    This processor takes as input a batch of observations and actions and returns a batch of tokens, patches,
    patch_positions, and loss masks and attention masks.

    ```python
    >>> processor = GiaProcessor(seq_len=15)
    >>> batch_data = processor(
    ...     continuous_observations=[[[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]],
    ...     discrete_actions=[[0, 1, 2]],
    ... )
    >>> batch_data["input_ids"]
    [[30512, 30632, 31024, 30000, 30665, 30685, 31024, 30001, 30699, 30710, 31024, 30002, None, None, None]]
    >>> batch_data["input_types"]
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None]]
    >>> batch_data["local_positions"]
    [[0, 1, None, None, 0, 1, None, None, 0, 1, None, None, None, None, None]]
    >>> batch_data["loss_mask"]
    [[False, False, True, None, False, False, True, None, False, False, True, None, None, None, None]]
    >>> batch_data["attention_mask"]
    [[True, True, True, True, True, True, True, True, True, True, True, True, False, False, False]]
    ```

    Args:
        mu (`float`, *optional*, defaults to `100`):
            The μ parameter for the μ-law companding of continuous observations and actions.
        M (`float`, *optional*, defaults to `256`):
            The M parameter for the μ-law companding of continuous observations and actions.
        nb_bins (`int`, *optional*, defaults to `1024`):
            The number of bins for the discretization of continuous observations and actions.
        patch_size (`int`, *optional*, defaults to `16`):
            The size of the patches to extract from images.
        mask_loss_modalities (`Union[List[str], str]`, *optional*, defaults to `"default"`):
            The modalities to mask for the loss computation. Defaults to all modalities except text and actions.
        seq_len (`int`, *optional*, defaults to `1024`):
            The length (number of tokens) of a sequence.
        local_positions_groups (`Union[List[List[str]], str]`, *optional*, defaults to `"default"`):
            The groups of modalities for which to add local positions. Defaults to a single group containing all
            observations modalities (text, images, discrete and continuous observations).
        use_separator (`bool`, *optional*, defaults to `True`):
            Whether to include a separator token between observations and actions.
    """

    features = datasets.Features(
        {
            "input_ids": datasets.Sequence(datasets.Value(dtype="int64")),
            "patches": datasets.Sequence(datasets.Image()),
            "patch_positions": datasets.Sequence(datasets.Array2D((2, 2), dtype="float32")),
            "input_types": datasets.Sequence(datasets.Value(dtype="int64")),
            "local_positions": datasets.Sequence(datasets.Value(dtype="int64")),
            "loss_mask": datasets.Sequence(datasets.Value(dtype="bool")),
            "attention_mask": datasets.Sequence(datasets.Value(dtype="bool")),
        }
    )

    def __init__(
        self,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        patch_size: int = 16,
        mask_loss_modalities: Union[List[str], str] = "default",
        seq_len: int = 1024,
        local_positions_groups: Union[List[List[str]], str] = "default",
        use_separator: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = GiaTokenizer(mu, M, nb_bins, patch_size)
        if mask_loss_modalities == "default":
            self.mask_loss_modalities = [
                # "text",
                "images",
                # "text_observations",
                "image_observations",
                "discrete_observations",
                "continuous_observations",
                # "discrete_actions",
                # "continuous_actions",
                "rewards",
            ]
        else:
            self.mask_loss_modalities = mask_loss_modalities
        if local_positions_groups == "default":
            local_positions_groups = [
                [
                    # "text",
                    # "images",
                    "text_observations",
                    "image_observations",
                    "discrete_observations",
                    "continuous_observations",
                    # "discrete_actions",
                    # "continuous_actions",
                    # "rewards",
                ]
            ]
        self.local_positions_adder = LocalPositionsAdder(local_positions_groups)
        self.use_separator = use_separator
        if use_separator:
            separator = {
                "input_ids": [self.tokenizer.vocab_size],
                "input_types": [0],
                "loss_mask": [True],
            }
        else:
            separator = None
        self.interleaver = Interleaver(separator)
        self.seq_len = seq_len

    @staticmethod
    def truncate_residual(
        batch_data: Dict[str, List[Optional[List[T]]]], max_len: int
    ) -> Dict[str, List[Optional[List[T]]]]:
        """
        Truncate input sequences into sub-sequences of length up to max_len. Any residual elements
        that don't reach max_len in length are used to form a new sub-sequence.

        Args:
            sequences (Dict[str, List[Optional[List[T]]]]): Sequences to truncate
            max_len (int): Maximum length of each sub-sequence

        Returns:
            Dict[str, List[Optional[List[T]]]]: Truncated sequences

        Example:
            >>> batch_data = {"a": [[0, 1, 2],  [3, 4, 5, 6]],
            ...              "b": [[7, 8, 9],  None]}
            >>> truncate_residual(batch_data, max_len=2)
            {"a": [[0, 1], [2], [3, 4], [5, 6]],
             "b": [[7, 8], [9], None, None]
        """
        truncated = {key: [] for key in batch_data}
        # Get the batch size
        batch_size = len(next(iter(batch_data.values())))
        for ep_idx in range(batch_size):
            # Get the length of the sequence (all sequences should have the same length)
            for key in batch_data:
                if batch_data[key][ep_idx] is not None:  # can be None
                    seq_len = len(batch_data[key][ep_idx])
                    break
            for start in range(0, seq_len, max_len):
                for key in batch_data:
                    sequence = batch_data[key][ep_idx]
                    if sequence is None:
                        truncated[key].append(None)
                    else:
                        subsequence = sequence[start : start + max_len]  # take a subsequence of max_len elements
                        truncated[key].append(subsequence)
        return truncated

    @staticmethod
    def pad_sequences(batch_data: Dict[str, List[T]], max_len: int) -> Tuple[List[T], List[int]]:
        """
        Pad sequences to a maximum length.

        Args:
            sequences (List[T]): A list of sequences.
            max_len (int): Maximum length for the output sequences.

        Returns:
            Tuple[List[T], List[int]]: A tuple of padded sequences and masks.
        """
        padded = {key: [] for key in batch_data}
        batch_size = len(next(iter(batch_data.values())))
        padded["attention_mask"] = []
        for seq_idx in range(batch_size):
            for key in batch_data:
                sequence = batch_data[key][seq_idx]
                if sequence is None:
                    padded[key].append(None)
                else:  # it's a list
                    if max_len < len(sequence):
                        raise RuntimeError(f"Sequence length {len(sequence)} is greater than max_len {max_len}.")
                    seq_len, pad_len = len(sequence), max_len - len(sequence)
                    sequence = sequence + [None] * pad_len
                    mask = [True] * seq_len + [False] * pad_len  # computed for every keys, but it's the same
                    padded[key].append(sequence)
            padded["attention_mask"].append(mask)
        return padded

    def __call__(
        self,
        text: Optional[str] = None,
        images: Optional[np.ndarray] = None,
        text_observations: NestedList[str] = None,
        image_observations: NestedList[Union[np.ndarray, Image.Image]] = None,
        discrete_observations: NestedList[int] = None,
        continuous_observations: NestedList[float] = None,
        discrete_actions: NestedList[int] = None,
        continuous_actions: NestedList[float] = None,
        rewards: NestedList[float] = None,
        interleave: bool = True,
        truncation: Union[bool, str] = "residual",
        truncation_side: str = "right",
        padding: Union[bool, str] = "max_length",
        max_length: Optional[int] = None,
    ):
        """
        Process input. Returns tokens, patches, patch_positions, and loss masks.

        Args:
            text (Optional[str], optional): Standalone text input. Defaults to None.
            images (Optional[np.ndarray], optional): Standalone image input. Defaults to None.
            text_observations (NestedList[str], optional): Episode text observations. Defaults to None.
            image_observations (NestedList[Union[np.ndarray, Image]], optional): Episode image observations.
                Defaults to None.
            discrete_observations (NestedList[int], optional): Episode discrete observations. Defaults to None.
            continuous_observations (NestedList[float], optional): Episode continuous observations. Defaults to None.
            discrete_actions (NestedList[int], optional): Episode discrete actions. Defaults to None.
            continuous_actions (NestedList[float], optional): Episode continuous actions. Defaults to None.
            rewards (NestedList[float], optional): Rewards. Defaults to None.
            interleave (bool, optional): Interleave observations and actions. Defaults to True.
            truncation (Union[bool, str]): Specifies the truncation strategy.
                - 'residual' (default): Truncate to a maximum length specified with `max_length` or to the maximum
                    acceptable input length for the model if `max_length` is not provided. Any residual elements that
                    don't reach `max_length` in length are used to form a new sub-sequence.
                - True or 'max_length': Truncate to a maximum length specified with `max_length` or to the maximum
                    acceptable input length for the model if `max_length` is not provided.
                - False or 'do_not_truncate': No truncation (i.e., can output a batch with sequences of different
                    lengths).
            truncation_side (str): Specifies the side to truncate when `truncation` is True or 'max_length'. Can be
                'left' or 'right' (default). With truncation='residual', this parameter can only be 'right'.
            padding (Union[bool, str]): Specifies the padding strategy.
                - True or 'longest': Pad to the length of the longest sequence in the batch (or no padding if only a
                    single sequence if provided).
                - 'max_length' (default): Pad to a maximum length specified with `max_length` or to the maximum
                    acceptable input length for the model if `max_length` is not provided.
                - False or 'do_not_pad': No padding (i.e., can output a batch with sequences of different
                    lengths).
            max_length (Optional[int]): Specifies the maximum length for padding and truncation. If not provided, the
                maximum acceptable input length for the model is used.

        Raises:
            ValueError: Invalid truncation strategy.
            ValueError: Invalid padding strategy.

        Returns:
            Dict[str, List[Any]]: A dictionary of tensors containing the tokenized inputs.
        """
        features = self.tokenizer(
            text,
            images,
            text_observations,
            image_observations,
            discrete_observations,
            continuous_observations,
            discrete_actions,
            continuous_actions,
            rewards,
        )

        # Add the loss mask
        for modality in features:
            if modality in self.mask_loss_modalities:
                features[modality]["loss_mask"] = nested_like(features[modality]["input_types"], False)

        # Add the local positions
        self.local_positions_adder(features)

        # Pop the reward, if any
        features.pop("rewards", None)

        if interleave:
            batch_data = self.interleaver(features)
        else:
            return features

        # Truncate sequences
        if truncation in [True, "max_length", "residual"]:
            max_length = max_length or self.seq_len
            if truncation == "residual":
                if truncation_side != "right":
                    raise ValueError("With truncation='residual', truncation_side can only be 'right'.")
                batch_data = self.truncate_residual(batch_data, max_len=max_length)
            else:  # True or "max_length"
                if truncation_side == "left":
                    batch_data = {key: [sequence[-max_length:] for sequence in batch_data[key]] for key in batch_data}
                elif truncation_side == "right":
                    batch_data = {key: [sequence[:max_length] for sequence in batch_data[key]] for key in batch_data}
        elif truncation in [False, "do_not_truncate"]:
            pass
        else:
            raise ValueError(f"Invalid truncation value: {truncation}")

        # Pad sequences
        if padding in [True, "longest", "max_length"]:
            if padding in [True, "longest"]:
                max_length = max(len(sequence) for sequence in batch_data["input_ids"])
            else:
                max_length = max_length or self.seq_len
            batch_data = self.pad_sequences(batch_data, max_length)
        elif padding in [False, "do_not_pad"]:
            pass
        else:
            raise ValueError(f"Invalid value for `padding`: {padding}")

        return batch_data
