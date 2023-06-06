from typing import Dict, List, Optional, Tuple, TypeVar, Union


import numpy as np
from PIL import Image


from gia.config import DatasetArguments
from .interleaver import Interleaver
from .tokenizer import GiaTokenizer
from .utils import nested_like
from .local_positions_adder import LocalPositionsAdder

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
NestedList = Union[None, T, List["NestedList[T]"]]


class GiaProcessor:
    def __init__(self, args: DatasetArguments) -> None:
        super().__init__()
        self.tokenizer = GiaTokenizer(args)
        args.text_vocab_size + args.nb_bins if args.use_separator else -1
        token_pad_value = 0
        local_position_pad_value = -1
        patch_pad_value = None
        patch_position_pad_value = None
        self.padding_value = {
            "input_ids": token_pad_value,
            "local_positions": local_position_pad_value,
            "patches": patch_pad_value,
            "patch_positions": patch_position_pad_value,
            "input_types": 0,
            "loss_mask": 0,
        }
        self.interleaver = Interleaver()
        self.seq_len = args.seq_len
        self.modality_to_mask = [
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
        self.local_positions_adder = LocalPositionsAdder(
            [["text_observations", "image_observations", "discrete_observations", "continuous_observations"]]
        )

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
    def pad_sequences(batch_data: Dict[str, List[T]], max_len: int, padding_value: T) -> Tuple[List[T], List[int]]:
        """
        Pad sequences with a padding value to a maximum length.

        Args:
            sequences (List[T]): A list of sequences.
            max_len (int): Maximum length for the output sequences.
            padding_value (T): Padding value.

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
                    sequence = sequence + [padding_value[key]] * pad_len
                    mask = [1] * seq_len + [0] * pad_len  # computed for every keys, but it's the same for all keys
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
            if modality in self.modality_to_mask:
                features[modality]["loss_mask"] = nested_like(features[modality]["input_types"], 0)

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
                batch_data = self.truncate_residual(batch_data, max_len=max_length)
            else:
                batch_data = [sequence[:max_length] for sequence in value]  # FIXME
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
            batch_data = self.pad_sequences(batch_data, max_length, self.padding_value)
        elif padding in [False, "do_not_pad"]:
            pass
        else:
            raise ValueError(f"Invalid value for `padding`: {padding}")

        return batch_data
