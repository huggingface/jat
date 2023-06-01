import math
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from PIL import Image
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
        if isinstance(x, (list, np.ndarray)):
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
    def extract_patches(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from an image.

        Args:
            image (Union[np.ndarray, Image.Image]): Image to extract patches from of shape (C, H, W).

        Returns:
            Tuple of:
                - patches (np.ndarray): Patches extracted from the image. Output has shape (N, 4, P, P), where P
                    is the patch size. Patches are flattened in row-major order.
                - patch_positions (np.ndarray): Relative position intervals of the patches. Output has shape
                    (N, 2, 2), where the last two dimensions are the start and end positions of the patch.
        """
        if isinstance(image, Image.Image):
            image = np.transpose(np.array(image), (2, 0, 1))
        P = self.patch_size
        C, H, W = image.shape
        # Reshape to the closest above multiple of the patch size
        # cv2 works with channels last, so we need to transpose the image.
        image = image.transpose(1, 2, 0)
        H = H - H % P + P if H % P != 0 else H
        W = W - W % P + P if W % P != 0 else W
        resized_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        image = resized_image.transpose(2, 0, 1)  # Back to channels first
        # Extract patches
        patches = image.reshape(C, H // P, P, W // P, P).transpose(1, 3, 0, 2, 4)
        patches = patches.reshape(-1, C, P, P)
        # Pad the image with 0 to have 4 channels
        pad_width = ((0, 4 - C), (0, 0), (0, 0))
        patches = [np.pad(patch, pad_width, mode="constant", constant_values=0) for patch in patches]
        # Compute the relative position intervals of the patches within the image
        # They are described as [[x_min, y_min], [x_max, y_max]]
        # Output shape is (N, 2, 2)
        patch_positions = np.array(
            [
                [[i / (H // P), j / (W // P)], [(i + 1) / (H // P), (j + 1) / (W // P)]]
                for i in range(H // P)
                for j in range(W // P)
            ],
            dtype=np.float32,
        ).tolist()

        return patches, patch_positions

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
    def decode_continuous(self, token: int) -> float:
        """
        Inverse of tokenize_continuous.

        Args:
            x (int): Token

        Returns:
            NestedFloatList: Continuous value
        """
        # Subtract token shift
        token = max(0, token - self.token_shift)

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
        images: Optional[np.ndarray] = None,
        text_observations: NestedList[str] = None,
        image_observations: NestedList[Union[np.ndarray, Image.Image]] = None,
        discrete_observations: NestedList[int] = None,
        continuous_observations: NestedList[float] = None,
        discrete_actions: NestedList[int] = None,
        continuous_actions: NestedList[float] = None,
        rewards: NestedList[float] = None,
    ) -> Dict[str, Dict[str, Union[NestedList[int], NestedList[np.ndarray]]]]:
        output = {}
        if text is not None:
            output["text"] = {"input_ids": self.tokenize_text(text)}

        if images is not None:
            patches, patch_positions = self.extract_patches(images)
            output["images"] = {"patches": patches, "patch_positions": patch_positions}

        if text_observations is not None:
            output["text_observations"] = {"input_ids": self.tokenize_text(text_observations)}

        if image_observations is not None:
            patches, patch_positions = self.extract_patches(image_observations)
            output["image_observations"] = {"patches": patches, "patch_positions": patch_positions}

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
        separator_token = args.text_vocab_size + args.nb_bins if args.use_separator else -1
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
        self.interleaver = Interleaver(
            separator_token, token_pad_value, local_position_pad_value, patch_pad_value, patch_position_pad_value
        )
        self.seq_len = args.seq_len

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
        tokens_and_patches = self.tokenizer(
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

        # Pop the reward, if any
        tokens_and_patches.pop("rewards", None)

        if interleave:
            batch_data = self.interleaver(tokens_and_patches)
        else:
            return tokens_and_patches

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
