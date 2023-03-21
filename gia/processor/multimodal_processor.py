from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from gia.utils.utils import discretize, inverse_mu_law, mu_law


class MultimodalProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # For each key, we need to check if it is text, image, continuous or discrete
        raise NotImplementedError


def is_text(x: Any) -> bool:
    """
    Check if input is text.

    It checks if the input a array of strings.
    """
    return all(isinstance(s, str) for s in x)


def is_image(x: Any) -> bool:
    """
    Check if input is an image.

    Returns True if the input is a 4D level nested list of integers.
    """
    return x.ndim == 4


def is_continuous(x: Any) -> bool:
    """
    Check if input is continous.

    Returns True if the input a list of float, or a list of list of float.
    """
    return x.dtype == np.float32


def is_discrete(x: Any) -> bool:
    """
    Check if input is discrete.

    Returns True if the input a list of integers, or a list of list of integers.
    """
    return x.dtype == np.int64


class MultimodalProcessor:
    """
    Multi-modal tokenizer.

    Example:
        >>> import numpy as np
        >>> tokenizer = MultimodalProcessor()
        >>> inputs = {
        ...     "texts": ["Go right", "Go left"],
        ...     "images": np.random.randint(0, 256, (2, 3, 16, 16), dtype=np.uint8).tolist(),
        ...     "continuous": [2.1, 3.4],
        ...     "actions": [[9, 8, 6], [5, 9, 9]],
        ... }
        >>> encoding = tokenizer(inputs)
        >>> encoding.keys()


    Args:
        mu (float, optional): Mu-law companding parameter. Defaults to 100.
        M (float, optional): Mu-law companding parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.
        token_shift (int, optional): Shift for the discrete tokens. Defaults to 32_000.
    """

    def __init__(
        self,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        token_shift: int = 32_000,
    ) -> None:
        super().__init__()
        self.token_shift = token_shift
        self.mu_law_compand = True
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.token_shift = token_shift
        # Token for separating observations and actions
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    def tokenize_discrete(self, x: List) -> List[List[int]]:
        # Unsqueeze when the input is a vector
        x = np.array(x, dtype=np.int64)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        tokens = x + self.token_shift
        return tokens

    def tokenize_continuous(self, x: List) -> List[List[int]]:
        # Normalize tensors to the range [-1, 1]
        x = np.array(x, dtype=np.float32)
        if self.mu_law_compand:
            x = mu_law(x, mu=self.mu, M=self.M)

        # Clip to the range [-1, 1]
        x = np.clip(x, -1.0, 1.0)

        # Discretize tensors
        x = discretize(x, nb_bins=self.nb_bins)

        # Unsqueeze when the input is a vector
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        # Unsqueeze if needed
        tokens = x + self.token_shift
        return tokens

    def inverse_tokenize_continuous(self, tokens: List) -> List:
        """
        Inverse tokenize continous.

        First, each integer element of input tensor is mapped to the center of the corresponding bin.
        Then, the tensor is de-mu-law companded if needed.

        Args:
            tokens (Tensor): Tokens

        Returns:
            Tensor: Reconstructed tensor
        """

        # Maps tokens from [0, nb_bins-1] to [-1, 1]
        # We map the bin number to the center of the bin
        x = (2 * tokens + 1) / self.nb_bins - 1

        # De-mu-law compand tensors
        if self.mu_law_compand:
            x = inverse_mu_law(x, mu=self.mu, M=self.M)

        return x

    def tokenize_text(self, x: List) -> List:
        tokens = self.text_tokenizer(x.tolist())
        return np.array(tokens["input_ids"])

    def __call__(self, inputs: Dict[str, List]) -> Dict[str, List]:
        output = {}
        for key in inputs:
            value = inputs[key]
            if is_text(value):
                output[key] = self.tokenize_text(value)
            elif is_image(value):  # Warning: special case for images:  make channels first
                value = np.array(value, dtype=np.uint8)
                if np.argmin(value.shape[1:]) == 2:  # channels last
                    value = np.transpose(value, (0, 3, 1, 2))
                assert np.argmin(value.shape[1:]) == 0, "Channels error"
                output[key] = value
            elif is_discrete(value):
                output[key] = self.tokenize_discrete(value)
            elif is_continuous(value):
                output[key] = self.tokenize_continuous(value)
            else:
                raise ValueError(f"Unknown input type for key '{key}'.")
        return output


if __name__ == "__main__":
    import numpy as np

    tokenizer = MultimodalProcessor()
    inputs = {
        "texts": ["Go right", "Go left"],
        "images": np.random.randint(0, 256, (2, 3, 16, 16), dtype=np.uint8),
        "continuous": [2.1, 3.4],
        "actions": [[9, 8, 6], [5, 9, 9]],
    }
    encoding = tokenizer(inputs)
    print(encoding)
