import math
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from transformers import AutoTokenizer

from .utils import nested_like


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
        >>> tokenizer = Tokenizer()
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

    def __init__(self, mu: float = 100.0, M: float = 256.0, nb_bins: int = 1024, patch_size: int = 16):
        super().__init__()
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.patch_size = patch_size

        self.mu_law_compand = True
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.token_shift = self.text_tokenizer.vocab_size

    @property
    def vocab_size(self) -> int:
        return self.text_tokenizer.vocab_size + self.nb_bins

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
            image = np.array(image)
        P = self.patch_size
        H, W, C = image.shape
        # Reshape to the closest above multiple of the patch size
        H = H - H % P + P if H % P != 0 else H
        W = W - W % P + P if W % P != 0 else W
        image = cv2.resize(
            image, (W, H), interpolation=cv2.INTER_AREA
        )  # cv2.resize expects (W, H), but actually outputs (H, W)
        # Extract patches
        shape = (H // P, W // P, P, P, C)
        strides = (P * image.strides[0], P * image.strides[1]) + image.strides
        patches = as_strided(image, shape=shape, strides=strides)
        patches = patches.reshape(-1, P, P, C)
        # Pad the image with 0 to have 4 channels
        pad_width = ((0, 0), (0, 0), (0, 4 - C))
        patches = [np.pad(patch, pad_width, mode="constant", constant_values=0) for patch in patches]
        # patches = [Image.fromarray(patche) for patche in patches]
        # Compute the relative position intervals of the patches within the image
        # They are described as [[x_min, y_min], [x_max, y_max]]
        # Output shape is (N, 2, 2)
        patch_positions = [
            np.array([[i / (H // P), j / (W // P)], [(i + 1) / (H // P), (j + 1) / (W // P)]], dtype=np.float32)
            for i in range(H // P)
            for j in range(W // P)
        ]

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
            input_ids = self.tokenize_text(text)
            input_types = nested_like(input_ids, 0)
            output["text"] = {"input_ids": input_ids, "input_types": input_types}

        if images is not None:
            patches, patch_positions = self.extract_patches(images)
            input_types = nested_like(patches, 1)
            output["images"] = {"patches": patches, "patch_positions": patch_positions, "input_types": input_types}

        if text_observations is not None:
            input_ids = self.tokenize_text(text_observations)
            input_types = nested_like(input_ids, 0)
            output["text_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if image_observations is not None:
            patches, patch_positions = self.extract_patches(image_observations)
            input_types = nested_like(patches, 1)
            output["image_observations"] = {
                "patches": patches,
                "patch_positions": patch_positions,
                "input_types": input_types,
            }

        if discrete_observations is not None:
            input_ids = self.tokenize_discrete(discrete_observations)
            input_types = nested_like(input_ids, 0)
            output["discrete_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if continuous_observations is not None:
            input_ids = self.tokenize_continuous(continuous_observations)
            input_types = nested_like(input_ids, 0)
            output["continuous_observations"] = {"input_ids": input_ids, "input_types": input_types}

        if discrete_actions is not None:
            input_ids = self.tokenize_discrete(discrete_actions)
            input_types = nested_like(input_ids, 0)
            output["discrete_actions"] = {"input_ids": input_ids, "input_types": input_types}

        if continuous_actions is not None:
            input_ids = self.tokenize_continuous(continuous_actions)
            input_types = nested_like(input_ids, 0)
            output["continuous_actions"] = {"input_ids": input_ids, "input_types": input_types}

        if rewards is not None:
            input_ids = self.tokenize_continuous(rewards)
            input_types = nested_like(input_ids, 0)
            output["rewards"] = {"input_ids": input_ids, "input_types": input_types}

        return output
