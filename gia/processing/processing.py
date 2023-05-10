import math
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from transformers import AutoTokenizer

from gia.config import DatasetArguments

from .interleaver import Interleaver, split_and_pad_sequences

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
        self.seq_len = args.seq_len

    def __call__(self, **kwargs):
        tokens_and_patches = self.tokenizer(**kwargs)
        # pop the reward, if any
        tokens_and_patches.pop("rewards", None)
        x = interleave_batch(tokens_and_patches)

        PATCH_PAD = np.zeros((3, 16, 16), dtype=np.int64)
        POSITION_PAD = [[0, 0], [0, 0]]

        padded_input_ids, masks_1 = split_and_pad_sequences(x["input_ids"], max_len=self.seq_len, pad_value=0)
        padded_patches, masks_2 = split_and_pad_sequences(
            x["patches"], max_len=self.seq_len, pad_value=PATCH_PAD
        )
        assert masks_1 == masks_2
        padded_positions, masks = split_and_pad_sequences(
            x["positions"], max_len=self.seq_len, pad_value=POSITION_PAD
        )
        padded_input_type, masks = split_and_pad_sequences(x["input_type"], max_len=self.seq_len, pad_value=0)

        x["input_ids"] = padded_input_ids
        x["patches"] = padded_patches
        x["positions"] = padded_positions
        x["input_type"] = padded_input_type
        x["attention_mask"] = masks
        return x


# def split_and_pad_sequences(sequences: List[List[Any]], max_len: int) -> Tuple[List[List[Any]], List[List[int]]]:
#     """
#     Splits input sequences into sub-sequences of length max_len and pads them if necessary.
#     Generates a mask indicating the padding positions.

#     Args:
#         sequences (List[List[int]]): A list of sequences, where each sequence is a list of integers.
#         max_len (int): Maximum length for the output sub-sequences.

#     Returns:
#         Tuple[List[List[int]], List[List[int]]]: A tuple containing two lists:
#             - The first list contains the padded sub-sequences.
#             - The second list contains the masks for the sub-sequences, with 1 indicating an original
#               element and 0 indicating a padded element.

#     Example:
#         >>> sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
#         >>> out, mask = split_and_pad_sequences(sequences, max_len=3)
#         >>> out
#         [[1, 2, 3], [4, 5, 4], [6, 7, 8], [9, 9, 9]]
#         >>> mask
#         [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]]
#     """
#     padded_subsequences = []
#     masks = []

#     for sequence in sequences:
#         for i in range(0, len(sequence), max_len):
#             # Take a subsequence of max_len elements
#             subsequence = sequence[i : i + max_len]
#             mask = [1] * len(subsequence)

#             # If the subsequence is smaller than max_len, pad it
#             if len(subsequence) < max_len:
#                 padding_length = max_len - len(subsequence)
#                 subsequence += [subsequence[0]] * padding_length
#                 mask += [0] * padding_length

#             padded_subsequences.append(subsequence)
#             masks.append(mask)

#     return padded_subsequences, masks


# class Padder:
#     """ """

#     def __call__(self, x: Dict) -> Any:
#         for mod, processed in x.items():
#             if isinstance(processed, dict):  # keys are input_ids, or patches and positions for images
#                 keys = list(processed.keys())  # to avoid modifying the dict while iterating over it
#                 for key in keys:
#                     if isinstance(processed[key], list) and isinstance(processed[key][0], list):
#                         # list of lists is actually a sequence
#                         out, mask = split_and_pad_sequences(processed[key], 3)
#                         x[mod][key] = out
#                     x[mod]["attention_mask"] = mask  # mask is the same for all keys
