import math
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from transformers import AutoTokenizer

from .interleaver import Interleaver
from .local_positions_adder import LocalPositionsAdder


ImageType = List[List[List[int]]]

T = TypeVar("T")


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def nested_like(x, val):
    if isinstance(x, list):
        return [nested_like(x_i, val) for x_i in x]
    else:
        return val


class GiaImageProcessor:
    def __init__(self, patch_size: int = 16) -> None:
        self.patch_size = patch_size

    def _resize_to_multiple_of_patch_size(self, image: np.ndarray) -> np.ndarray:
        P = self.patch_size
        H, W, _ = image.shape
        # Resize to the closest above multiple of the patch size
        H = H - H % P + P if H % P != 0 else H
        W = W - W % P + P if W % P != 0 else W
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        return image

    def _extract_patches(self, image: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        image = self._resize_to_multiple_of_patch_size(image)
        # Pad the image with 0 to have 4 channels
        image = np.pad(image, ((0, 0), (0, 0), (0, 4 - image.shape[2])), mode="constant", constant_values=0)
        # Get the number of patches per row and column
        num_patches_per_col = image.shape[0] // self.patch_size
        num_patches_per_row = image.shape[1] // self.patch_size
        # Extract patches
        shape = (num_patches_per_col, num_patches_per_row, self.patch_size, self.patch_size, 4)
        strides = (self.patch_size * image.strides[0], self.patch_size * image.strides[1]) + image.strides
        patches = as_strided(image, shape=shape, strides=strides)
        patches = patches.reshape(-1, self.patch_size, self.patch_size, 4)
        # Compute the relative position intervals of the patches within the image
        # They are described as [[x_min, y_min], [x_max, y_max]]
        # Output shape is (N, 2, 2) with N the total number of patches in the image
        patch_positions = [
            [
                [col / (num_patches_per_col), row / (num_patches_per_row)],
                [(col + 1) / (num_patches_per_col), (row + 1) / (num_patches_per_row)],
            ]
            for col in range(num_patches_per_col)
            for row in range(num_patches_per_row)
        ]
        # To channels first
        patches = [patch.transpose(2, 0, 1) for patch in patches]
        return patches, patch_positions

    def __call__(self, images: List[ImageType]):
        output = {"patches": [], "patch_positions": []}
        for image in images:
            patches, patch_positions = self._extract_patches(np.array(image, dtype=np.uint8))
            output["patches"].append(patches)
            output["patch_positions"].append(patch_positions)
        return output


class GiaContinuousTokenizer:
    def __init__(self, mu: float = 100.0, M: float = 256.0, nb_bins: int = 1024, token_shift: int = 0) -> None:
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.token_shift = token_shift
        self.mu_law_compand = True

    @property
    def vocab_size(self) -> int:
        return self.nb_bins

    def _float_to_token(self, x: float) -> int:
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

    def _token_to_float(self, token: int) -> float:
        # Subtract token shift
        token = max(0, token - self.token_shift)
        # Maps tokens from [0, nb_bins-1] to [-1, 1]; We map the bin number to the center of the bin
        val = (2 * token + 1) / self.nb_bins - 1
        # De-mu-law compand tensors
        if self.mu_law_compand:
            val = math.copysign(1, val) * (math.exp(abs(val) * math.log(self.M * self.mu + 1.0)) - 1.0) / self.mu
        return val

    def decode(self, tokens: List[List[int]]) -> List[List[float]]:
        return [[self._token_to_float(token_ij) for token_ij in token_i] for token_i in tokens]

    def __call__(self, x: List[List[float]]):
        input_ids = [[self._float_to_token(x_ij) for x_ij in x_i] for x_i in x]
        return {"input_ids": input_ids}


class GiaDiscreteTokenizer:
    def __init__(self, max_value: int = 1024, token_shift: int = 0) -> None:
        self.max_value = max_value
        self.token_shift = token_shift

    def decode(self, tokens: List[List[int]]) -> Union[List[int], List[List[int]]]:
        return [[clamp(token_ij - self.token_shift, 0, self.max_value) for token_ij in token_i] for token_i in tokens]

    def __call__(self, x: Union[List[int], List[List[int]]]):
        input_ids = [[clamp(x_ij, 0, self.max_value) + self.token_shift for x_ij in x_i] for x_i in x]
        return {"input_ids": input_ids}


class GiaProcessor:
    r"""
    Processor for Gia.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to extract from image observations.
        text_tokenizer_name (`str`, *optional*, default to `"albert-base-v2"`):
            Name of the pretrained tokenizer for text to use.
        mu (`float`, *optional*, default to 100.0):
            μ parameter for the μ-law companding of continuous observations and actions.
        M (`float`, *optional*, default to 256.0):
            M parameter for the μ-law companding of continuous observations and actions.
        nb_bins (`int`, *optional*, defaults to 1024):
            Number of bins for the discretization of continuous observations and actions. It's also used as the max
            value for discrete tokenizer.
        mask_loss_modalities (`str` or `List[str]`, *optional*, default to `"default"`):
            Modalities to mask for the loss computation. Defaults to all modalities except text and actions.
        seq_len (`int`, *optional*, default to 2048):
            The length (number of tokens) of a sequence.
        local_positions_group (`str` or `List[List[str]]`, *optional*, default to `default`):
            The groups of modalities for which to add local positions. Defaults to a single group containing all
            observations modalities (text, images, discrete and continuous observations).
        use_separator (`bool`, *optional*, default to `True`):
            Whether to include a separator token between observations and actions.
    """

    def __init__(
        self,
        patch_size: int = 16,
        text_tokenizer_name: str = "albert-base-v2",
        mu: float = 100.0,
        M: float = 256.0,
        nb_bins: int = 1024,
        mask_loss_modalities: Union[str, List[str]] = "default",
        seq_len: int = 2048,
        local_positions_groups: Union[str, List[List[str]]] = "default",
        use_separator: bool = True,
    ):
        self.image_processor = GiaImageProcessor(patch_size)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        token_shift = self.text_tokenizer.vocab_size
        self.continuous_tokenizer = GiaContinuousTokenizer(mu, M, nb_bins, token_shift)
        self.discrete_tokenizer = GiaDiscreteTokenizer(max_value=nb_bins, token_shift=token_shift)

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
                "input_ids": [token_shift + nb_bins],
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

    def decode_discrete(self, tokens: List[List[int]]) -> List[List[int]]:
        return self.discrete_tokenizer.decode(tokens)

    def decode_continuous(self, tokens: List[List[int]]) -> List[List[float]]:
        return self.continuous_tokenizer.decode(tokens)

    def __call__(
        self,
        text: Optional[List[str]] = None,
        images: Optional[List[ImageType]] = None,
        text_observations: Optional[List[List[str]]] = None,
        image_observations: Optional[List[List[ImageType]]] = None,
        discrete_observations: Optional[List[List[List[int]]]] = None,
        continuous_observations: Optional[List[List[List[float]]]] = None,
        discrete_actions: Optional[List[List[List[int]]]] = None,
        continuous_actions: Optional[List[List[List[float]]]] = None,
        rewards: Optional[List[List[float]]] = None,
        interleave: bool = True,
        truncation: Union[bool, str] = "residual",
        truncation_side: str = "right",
        padding: Union[bool, str] = "max_length",
        max_length: Optional[int] = None,
    ):
        batch_encoding = {}
        if text is not None:
            input_ids = self.text_tokenizer(text)["input_ids"]
            input_types = [[0] * len(val) for val in input_ids]
            batch_encoding["text"] = {"input_ids": input_ids, "input_types": input_types}

        if images is not None:
            output = self.image_processor(images)
            patches = output["patches"]
            patch_positions = output["patch_positions"]
            input_types = [[1] * len(val) for val in patches]
            batch_encoding["images"] = {
                "patches": patches,
                "patch_positions": patch_positions,
                "input_types": input_types,
            }

        if text_observations is not None:
            input_ids = [self.text_tokenizer(seq)["input_ids"] for seq in text_observations]
            input_types = [[[0] * len(val) for val in seq] for seq in input_ids]
            batch_encoding["text_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if image_observations is not None:
            output = [self.image_processor(seq) for seq in image_observations]
            patches = [ep["patches"] for ep in output]
            patch_positions = [ep["patch_positions"] for ep in output]
            input_types = [[[1] * len(val) for val in seq] for seq in patches]
            batch_encoding["image_observations"] = {
                "patches": patches,
                "patch_positions": patch_positions,
                "input_types": input_types,
            }

        if discrete_observations is not None:
            input_ids = [self.discrete_tokenizer(ep)["input_ids"] for ep in discrete_observations]
            input_types = [[[0] * len(val) for val in seq] for seq in input_ids]
            batch_encoding["discrete_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if continuous_observations is not None:
            input_ids = [self.continuous_tokenizer(ep)["input_ids"] for ep in continuous_observations]
            input_types = [[[0] * len(val) for val in seq] for seq in input_ids]
            batch_encoding["continuous_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if discrete_actions is not None:
            discrete_actions = [[[action] for action in seq] for seq in discrete_actions]
            input_ids = [self.discrete_tokenizer(ep)["input_ids"] for ep in discrete_actions]
            input_types = [[[0] for _ in seq] for seq in input_ids]
            batch_encoding["discrete_actions"] = {"input_ids": input_ids, "input_types": input_types}

        if continuous_actions is not None:
            input_ids = [self.continuous_tokenizer(ep)["input_ids"] for ep in continuous_actions]
            input_types = [[[0] * len(val) for val in seq] for seq in input_ids]
            batch_encoding["continuous_actions"] = {"input_ids": input_ids, "input_types": input_types}

        if rewards is not None:
            rewards = [[[reward] for reward in seq] for seq in rewards]
            input_ids = [self.continuous_tokenizer(ep)["input_ids"] for ep in rewards]
            input_types = [[[0] for _ in seq] for seq in input_ids]
            batch_encoding["rewards"] = {"input_ids": input_ids, "input_types": input_types}

        # Add the loss mask
        for modality in batch_encoding:
            if modality in self.mask_loss_modalities:
                batch_encoding[modality]["loss_mask"] = nested_like(batch_encoding[modality]["input_types"], False)

        # Add the local positions
        self.local_positions_adder(batch_encoding)

        # Pop the reward, if any
        batch_encoding.pop("rewards", None)

        if interleave:
            batch_data = self.interleaver(batch_encoding)
        else:
            return batch_encoding

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
