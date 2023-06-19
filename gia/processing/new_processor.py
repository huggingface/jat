import math
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from transformers import BatchEncoding, PreTrainedTokenizer, image_utils
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.processing_utils import ProcessorMixin

from .interleaver import Interleaver
from .local_positions_adder import LocalPositionsAdder
from .utils import nested_like


ImageInput = Union["Image.Image", np.ndarray, List["Image.Image"], List[np.ndarray]]

T = TypeVar("T")


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


class GiaImageProcessor(BaseImageProcessor):
    def __init__(self, patch_size: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
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

    def _to_numpy_array(self, images: ImageInput) -> np.ndarray:
        # From every possible type of input, convert to a numpy array or to a list of numpy arrays (if batched)
        if isinstance(images, list):
            if all(isinstance(image, np.ndarray) for image in images):
                return images
            elif all(isinstance(image, Image.Image) for image in images):
                return [np.array(image) for image in images]
            else:
                raise TypeError("images must be a list of numpy arrays or a list of PIL images.")
        elif isinstance(images, np.ndarray):
            return images  # single image
        elif isinstance(images, Image.Image):
            return np.array(images)
        else:
            raise TypeError("images must be a numpy array or a PIL image.")

    def __call__(self, images: image_utils.ImageInput) -> BatchFeature:
        images = self._to_numpy_array(images)
        is_batched = isinstance(images, list)
        if is_batched:
            output = {"patches": [], "patch_positions": []}
            for image in images:
                patches, patch_positions = self._extract_patches(self._to_numpy_array(image))
                output["patches"].append(patches)
                output["patch_positions"].append(patch_positions)
        else:
            patches, patch_positions = self._extract_patches(self._to_numy_array(images))
            output = {"patches": patches, "patch_positions": patch_positions}
        return BatchFeature(output)


GiaImageProcessor.register_for_auto_class()


class GiaContinuousTokenizer(PreTrainedTokenizer):
    def __init__(
        self, mu: float = 100.0, M: float = 256.0, nb_bins: int = 1024, token_shift: int = 0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.token_shift = token_shift
        self.mu_law_compand = True

    @property
    def vocab_size(self) -> int:
        return self.nb_bins

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

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

    def decode(self, tokens: Union[List[int], List[List[int]]]) -> Union[List[float], List[List[float]]]:
        if isinstance(tokens, List) and all(isinstance(token, int) for token in tokens):
            return [self._token_to_float(token) for token in tokens]
        elif isinstance(tokens, List) and all(isinstance(token, List) for token in tokens):
            return [[self._token_to_float(token_ij) for token_ij in token_i] for token_i in tokens]
        else:
            raise TypeError("tokens must be a list of integers or a list of list of integers.")

    def __call__(self, x: Union[List[float], List[List[float]]]) -> BatchEncoding:
        if isinstance(x, List) and all(isinstance(x_i, float) for x_i in x):
            input_ids = [self._float_to_token(x_i) for x_i in x]
        elif isinstance(x, List) and all(isinstance(x_i, List) for x_i in x):
            input_ids = [[self._float_to_token(x_ij) for x_ij in x_i] for x_i in x]
        else:
            raise TypeError("x must be a list of floats or a list of list of floats.")
        return BatchEncoding({"input_ids": input_ids})


GiaContinuousTokenizer.register_for_auto_class()


class GiaDiscreteTokenizer(PreTrainedTokenizer):
    def __init__(self, max_value: int = 1024, token_shift: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_value = max_value
        self.token_shift = token_shift

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

    def decode(self, tokens: Union[List[int], List[List[int]]]) -> Union[List[int], List[List[int]]]:
        if isinstance(tokens, List) and all(isinstance(token, int) for token in tokens):
            return [clamp(token - self.token_shift, 0, self.max_value) for token in tokens]
        elif isinstance(tokens, List) and all(isinstance(token, List) for token in tokens):
            return [
                [clamp(token_ij - self.token_shift, 0, self.max_value) for token_ij in token_i] for token_i in tokens
            ]
        else:
            raise TypeError("tokens must be a list of integers or a list of list of integers.")

    def __call__(self, x: Union[List[int], List[List[int]]]) -> BatchEncoding:
        if isinstance(x, List) and all(isinstance(x_i, int) for x_i in x):
            input_ids = [clamp(x_i, 0, self.max_value) + self.token_shift for x_i in x]
        elif isinstance(x, List) and all(isinstance(x_i, List) for x_i in x):
            input_ids = [[clamp(x_ij, 0, self.max_value) + self.token_shift for x_ij in x_i] for x_i in x]
        else:
            raise TypeError("x must be a list of ints or a list of list of ints.")
        return BatchEncoding({"input_ids": input_ids})


GiaDiscreteTokenizer.register_for_auto_class()


class GiaProcessor(ProcessorMixin):
    r"""
    Constructs an OneFormer processor which wraps [`OneFormerImageProcessor`] and
    [`CLIPTokenizer`]/[`CLIPTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities.

    Args:
        image_processor ([`TODO`]):
            The image processor.
        text_tokenizer ([`TODO`]):
            The tokenizer for text.
        continuous_tokenizer ([`TODO`]):
            The tokenizer for continuous values.
        discrete_tokenizer ([`TODO`]):
            The tokenizer for discrete values.
    """
    attributes = [
        "image_processor",
        "text_tokenizer",
        "continuous_tokenizer",
        "discrete_tokenizer",
    ]
    image_processor_class = "AutoImageProcessor"
    text_tokenizer_class = "AutoTokenizer"
    continuous_tokenizer_class = "AutoTokenizer"
    discrete_tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        text_tokenizer=None,
        continuous_tokenizer=None,
        discrete_tokenizer=None,
        mask_loss_modalities: Union[List[str], str] = "default",
        seq_len: int = 1024,
        local_positions_groups: Union[List[List[str]], str] = "default",
        use_separator: bool = True,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if text_tokenizer is None:
            raise ValueError("You need to specify a `text_tokenizer`.")
        if continuous_tokenizer is None:
            raise ValueError("You need to specify a `continuous_tokenizer`.")
        if discrete_tokenizer is None:
            raise ValueError("You need to specify a `discrete_tokenizer`.")

        if continuous_tokenizer.token_shift != text_tokenizer.vocab_size:
            raise ValueError(
                f"The continuous tokenizer must have a token shift (currently {continuous_tokenizer.token_shift}) "
                f"equal to the vocab size of the text tokenizer (currently {text_tokenizer.vocab_size})."
            )
        if discrete_tokenizer.token_shift != text_tokenizer.vocab_size:
            raise ValueError(
                "The discrete tokenizer must have a token shift equal to the vocab size of the text tokenizer."
            )
        super().__init__(
            image_processor,
            text_tokenizer,
            continuous_tokenizer,
            discrete_tokenizer,
            **kwargs,
        )

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
        if use_separator:
            separator = {
                "input_ids": [continuous_tokenizer.token_shift + continuous_tokenizer.vocab_size + 1],
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

    @property
    def vocab_size(self) -> int:
        return self.text_tokenizer.vocab_size + self.continuous_tokenizer.vocab_size

    def _fix_discrete_actions(
        self,
        discrete_actions: Union[int, List[int], List[List[int]]],
        rewards: Optional[Union[float, List[float]]] = None,
    ) -> Union[List[int], List[List[int]]]:
        # The purpose of this section is to "fix" the format of 'discrete_actions'.
        # The thing is, when it's a list of ints, we don't know if it's batched or not.
        # We need to infer it from 'rewards'.
        # If 'rewards' is a list of floats, then 'discrete_actions' is batched, and we need to unsqueeze it.
        # If 'rewards' is a float, then 'discrete_actions' is not batched, and we can leave it as it is.

        if isinstance(discrete_actions, List) and all(isinstance(action, int) for action in discrete_actions):
            if isinstance(rewards, List):
                return [[action] for action in discrete_actions]
            elif isinstance(rewards, float):
                return discrete_actions
            else:
                raise RuntimeError("Failed to infer whether the input is batched or not.")
        else:
            return discrete_actions

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        text_observations: Optional[Union[str, List[str]]] = None,
        image_observations: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        discrete_observations: Optional[Union[List[int], List[List[int]]]] = None,
        continuous_observations: Optional[Union[List[float], List[List[float]]]] = None,
        discrete_actions: Optional[Union[int, List[int], List[List[int]]]] = None,  # pa case here, we allow int
        continuous_actions: Optional[Union[List[float], List[List[float]]]] = None,
        rewards: Optional[Union[float, List[float]]] = None,
        interleave: bool = True,
        truncation: Union[bool, str] = "residual",
        truncation_side: str = "right",
        padding: Union[bool, str] = "max_length",
        max_length: Optional[int] = None,
    ) -> BatchEncoding:
        discrete_actions = self._fix_discrete_actions(discrete_actions, rewards)

        batch_encoding = {}
        if text is not None:
            input_ids = self.text_tokenizer(text)["input_ids"]
            input_types = nested_like(input_ids, 0)
            batch_encoding["text"] = {"input_ids": input_ids, "input_types": input_types}

        if images is not None:
            patches, patch_positions = self.image_processor(images)
            input_types = nested_like(patches, 1)
            batch_encoding["images"] = {
                "patches": patches,
                "patch_positions": patch_positions,
                "input_types": input_types,
            }

        if text_observations is not None:
            input_ids = self.text_tokenizer(text_observations)["input_ids"]
            input_types = nested_like(input_ids, 0)
            batch_encoding["text_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if image_observations is not None:
            patches, patch_positions = self.image_processor(images)
            input_types = nested_like(patches, 1)
            batch_encoding["images_observations"] = {
                "patches": patches,
                "patch_positions": patch_positions,
                "input_types": input_types,
            }

        if discrete_observations is not None:
            input_ids = self.discrete_tokenizer(discrete_observations)
            input_types = nested_like(input_ids, 0)
            batch_encoding["discrete_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if continuous_observations is not None:
            input_ids = self.continuous_tokenizer(continuous_observations)
            input_types = nested_like(input_ids, 0)
            batch_encoding["continuous_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if discrete_actions is not None:
            input_ids = self.discrete_tokenizer(discrete_actions)
            input_types = nested_like(input_ids, 0)
            batch_encoding["discrete_actions"] = {"input_ids": input_ids, "input_types": input_types}

        if continuous_actions is not None:
            input_ids = self.continuous_tokenizer(continuous_actions)
            input_types = nested_like(input_ids, 0)
            batch_encoding["continuous_actions"] = {"input_ids": input_ids, "input_types": input_types}

        if rewards is not None:
            input_ids = self.continuous_tokenizer(rewards)
            input_types = nested_like(input_ids, 0)
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
