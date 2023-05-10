import math
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from transformers import AutoTokenizer

from gia.config import DatasetArguments

from .interleaver import Interleaver

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
NestedList = Union[None, T, List["NestedList[T]"]]


def nested_decorator(func: Callable[[T], U]) -> Callable[[NestedList[T]], NestedList[U]]:
    @wraps(func)
    def wrapper(self, x):
        if x is None:
            return None
        if isinstance(x, list):
            return [wrapper(self, x_i) for x_i in x]
        else:
            return func(self, x)

    return wrapper


def tuple_nested_decorator(
    func: Callable[[T], Tuple[U, V]]
) -> Callable[[NestedList[T]], Tuple[NestedList[U], NestedList[V]]]:
    @wraps(func)
    def wrapper(self, x):
        if x is None:
            return None, None
        elif isinstance(x, list):
            output = [wrapper(self, x_i) for x_i in x]
            return (list(val) for val in zip(*output))
        else:
            return func(self, x)

    return wrapper


class GiaTokenizer:
    """
    Tokenizer for the Gia model.

    Args:
        args (:obj:`DatasetArguments`): Dataset arguments.

    Example:
        >>> from gia.config import DatasetArguments
        >>> from gia.processing import Tokenizer
        >>> args = DatasetArguments()
        >>> tokenizer = Tokenizer(args)
        >>> inputs = {
        ...     "text_observations": ["Go right", "Go left"],
        ...     "continuous_observations": [[0.1, 0.2], [0.3, 0.4]],
        ...     "discrete_actions": [1, 2],
        ... }
        >>> tokenizer(**inputs)
        {'text_observations': [[2, 162, 193, 3], [2, 162, 225, 3]],
         'continuous_observations': [[30632, 30665], [30685, 30699]],
         'discrete_actions': [30001, 30002]}
    """

    def __init__(self, args: DatasetArguments) -> None:
        super().__init__()
        self.mu = args.mu
        self.M = args.M
        self.nb_bins = args.nb_bins
        self.patch_size = args.patch_size
        self.seq_len = args.seq_len

        self.mu_law_compand = True
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.token_shift = self.text_tokenizer.vocab_size
        self.vocab_size = self.token_shift + self.nb_bins

    @nested_decorator
    def tokenize_text(self, text: str) -> int:
        output = self.text_tokenizer(text)
        return output["input_ids"]

    @tuple_nested_decorator
    def extract_patches(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from images.

        Args:
            image (np.ndarray): Image to extract patches from of shape (C, H, W).

        Returns:
            Tuple of:
                - patches (np.ndarray): Patches extracted from the image. Output has shape (N, C, P, P), where P
                    is the patch size. Patches are flattened in row-major order.
                - positions (np.ndarray): Relative position intervals of the patches. Output has shape
                    (N, 2, 2), where the last two dimensions are the start and end positions of the patch.
        """
        P = self.patch_size
        C, H, W = image.shape
        # First, reshape to the closest above multiple of the patch size
        # cv2 works with channels last, so we need to transpose the image.
        image = image.transpose(1, 2, 0)
        H = H - H % P + P if H % P != 0 else H
        W = W - W % P + P if W % P != 0 else W
        resized_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        image = resized_image.transpose(2, 0, 1)  # Back to channels first
        patches = image.reshape(C, H // P, P, W // P, P).transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(-1, C, P, P)
        # relative position intervals of the patches within the image
        # described with array [[x_min, y_min], [x_max, y_max]]
        # Output shape is (N, 2, 2)
        positions = np.array(
            [
                [[i / (H // P), j / (W // P)], [(i + 1) / (H // P), (j + 1) / (W // P)]]
                for i in range(H // P)
                for j in range(W // P)
            ],
            dtype=np.float32,
        ).tolist()

        return patches, positions

    @nested_decorator
    def tokenize_discrete(self, x: int) -> int:
        token = x + self.token_shift
        return token

    @nested_decorator
    def tokenize_continuous(self, x: float) -> int:
        # Normalize tensors to the range [-1, 1]
        if self.mu_law_compand:
            x = math.copysign(1, x) * math.log(self.mu * abs(x) + 1.0) / math.log(self.M * self.mu + 1.0)

        # Clip to the range [-1, 1]
        x = max(min(x, 1), -1)

        # Discretize tensors
        x = (x + 1.0) / 2 * self.nb_bins  # [-1, 1] to [0, nb_bins]
        x = math.floor(x)
        x = self.nb_bins - 1 if x == self.nb_bins else x  # Handle the case where x == 1.0

        # Shift
        token = x + self.token_shift
        return token

    @nested_decorator
    def inverse_tokenize_continuous(self, token: int) -> float:
        """
        Inverse of tokenize_continuous.

        Args:
            x (int): Token

        Returns:
            NestedFloatList: Continuous value
        """
        # Subtract token shift
        token = token - self.token_shift

        # Maps tokens from [0, nb_bins-1] to [-1, 1]
        # We map the bin number to the center of the bin
        val = (2 * token + 1) / self.nb_bins - 1

        # De-mu-law compand tensors
        if self.mu_law_compand:
            val = math.copysign(1, val) * (math.exp(abs(val) * math.log(self.M * self.mu + 1.0)) - 1.0) / self.mu

        return val

    def __call__(
        self,
        text: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        text_observations: NestedList[str] = None,
        image_observations: NestedList[np.ndarray] = None,
        discrete_observations: NestedList[int] = None,
        continuous_observations: NestedList[float] = None,
        discrete_actions: NestedList[int] = None,
        continuous_actions: NestedList[float] = None,
        rewards: NestedList[float] = None,
    ) -> Dict[str, Dict[str, Union[NestedList[int], NestedList[np.ndarray]]]]:
        output = {}
        if text is not None:
            output["text"] = {"input_ids": self.tokenize_text(text)}

        if image is not None:
            patches, positions = self.extract_patches(image)
            output["image"] = {"patches": patches, "positions": positions}

        if text_observations is not None:
            output["text_observations"] = {"input_ids": self.tokenize_text(text_observations)}

        if image_observations is not None:
            patches, positions = self.extract_patches(image_observations)
            output["image_observations"] = {"patches": patches, "positions": positions}

        if discrete_observations is not None:
            output["discrete_observations"] = {"input_ids": self.tokenize_discrete(discrete_observations)}

        if continuous_observations is not None:
            output["continuous_observations"] = {"input_ids": self.tokenize_continuous(continuous_observations)}

        if discrete_actions is not None:
            output["discrete_actions"] = {"input_ids": self.tokenize_discrete(discrete_actions)}

        if continuous_actions is not None:
            output["continuous_actions"] = {"input_ids": self.tokenize_continuous(continuous_actions)}

        if rewards is not None:
            output["rewards"] = {"input_ids": self.tokenize_continuous(rewards)}

        return output


class GiaProcessor:
    def __init__(self, args: DatasetArguments) -> None:
        super().__init__()
        self.tokenizer = GiaTokenizer(args)
        self.interleaver = Interleaver()
        self.seq_len = args.seq_len

        self.padding_value = {
            "input_ids": 0,
            "patches": np.zeros((4, args.patch_size, args.patch_size), dtype=np.uint8),
            "positions": [[0.0, 0.0], [0.0, 0.0]],
            "input_type": 0,
            "loss_mask": 0,
        }

    @staticmethod
    def truncate_residual(sequences: List[List[T]], max_len: int) -> List[List[T]]:
        """
        Truncate input sequences into sub-sequences of length up to max_len. Any residual elements
        that don't reach max_len in length are used to form a new sub-sequence.

        Args:
            sequences (List[List[T]]): A list of sequences, where each sequence is a list of items.
            max_len (int): Maximum length for the output sub-sequences.

        Returns:
            List[List[T]]: A list of truncated subsequences. Each subsequence has a length of up to max_len.
                           If the original sequence doesn't evenly divide by max_len, the last subsequence
                           will contain the remaining elements.

        Example:
            >>> sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
            >>> truncate_residual(sequences, max_len=3)
            [[1, 2, 3], [4, 5], [6, 7, 8], [9]]
        """
        truncated = []
        for sequence in sequences:
            for i in range(0, len(sequence), max_len):
                # Take a subsequence of max_len elements
                subsequence = sequence[i : i + max_len]
                truncated.append(subsequence)
        return truncated

    @staticmethod
    def pad_sequences(sequences: List[T], max_len: int, padding_value: T) -> Tuple[List[T], List[int]]:
        """
        Pad sequences with a padding value to a maximum length.

        Args:
            sequences (List[T]): A list of sequences.
            max_len (int): Maximum length for the output sequences.
            padding_value (T): Padding value.

        Returns:
            Tuple[List[T], List[int]]: A tuple of padded sequences and masks.
        """
        padded = []
        masks = []
        for sequence in sequences:
            if max_len < len(sequence):
                raise RuntimeError(f"Sequence length {len(sequence)} is greater than max_len {max_len}.")
            seq_len, pad_len = len(sequence), max_len - len(sequence)
            sequence = sequence + [padding_value] * pad_len
            mask = [1] * seq_len + [0] * pad_len
            padded.append(sequence)
            masks.append(mask)
        return padded, masks

    def __call__(
        self,
        text: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        text_observations: NestedList[str] = None,
        image_observations: NestedList[np.ndarray] = None,
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
        Process input. Returns tokens, patches, positions, and loss masks.

        Args:
            text (Optional[str], optional): Standalone text input. Defaults to None.
            image (Optional[np.ndarray], optional): Standalone image input. Defaults to None.
            text_observations (NestedList[str], optional): Episode text observations. Defaults to None.
            image_observations (NestedList[np.ndarray], optional): Episode image observations. Defaults to None.
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

        tokens_and_patches = self.tokenizer(
            text,
            image,
            text_observations,
            image_observations,
            discrete_observations,
            continuous_observations,
            discrete_actions,
            continuous_actions,
            rewards,
        )

        # Pop the reward, if any
        tokens_and_patches.pop("rewards", None)

        if interleave:
            batch_data = self.interleaver(tokens_and_patches)
        else:
            return tokens_and_patches

        # Truncate sequences
        if truncation in [True, "max_length", "residual"]:
            max_length = max_length or self.seq_len
            for key, value in batch_data.items():
                if truncation == "residual":
                    batch_data[key] = self.truncate_residual(value, max_len=max_length)
                else:
                    batch_data[key] = [sequence[:max_length] for sequence in value]
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

            for key, value in batch_data.items():
                padding_value = self.padding_value[key]
                batch_data[key], mask = self.pad_sequences(value, max_length, padding_value)
            batch_data["attention_mask"] = mask
        elif padding in [False, "do_not_pad"]:
            pass
        else:
            raise ValueError(f"Invalid value for `padding`: {padding}")

        return batch_data
