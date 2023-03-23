from typing import Dict, Iterable

import cv2
import numpy as np
from transformers import AutoTokenizer

from gia.utils.utils import discretize, inverse_mu_law, mu_law


def is_text(x: Iterable) -> bool:
    """
    Check if input is text.

    It checks if the input a array of strings.
    """
    return all(isinstance(s, str) for s in x)


def is_image(x: np.ndarray) -> bool:
    """
    Check if input is an image.

    Returns True if the input has 4 dimensions.
    """
    return x.ndim == 4


def is_continuous(x: np.ndarray) -> bool:
    """
    Check if input is continous.

    Returns True if the dtype is float32 or float64.
    """
    return x.dtype in [np.float32, np.float64]


def is_discrete(x: np.ndarray) -> bool:
    """
    Check if input is discrete.

    Returns True if the dtype is int64.
    """
    return x.dtype == np.int64


class MultimodalProcessor:
    """
    Multi-modal tokenizer.

    Example:
        >>> import numpy as np
        >>> tokenizer = MultimodalProcessor()
        >>> inputs = {
        ...     "texts": np.array(["Go right", "Go left"]),
        ...     "images": np.random.randint(0, 256, (2, 3, 16, 16), dtype=np.uint8),
        ...     "continuous": np.array([2.1, 3.4]),
        ...     "actions": np.array([[9, 8, 6], [5, 9, 9]]),
        ... }
        >>> encoding = tokenizer(inputs)
        >>> encoding.keys()
        dict_keys(['texts', 'images', '_positions/images', 'continuous', 'actions'])
        >>> encoding["texts"].shape
        (2, 4)
        >>> encoding["images"].shape
        (2, 1, 3, 16, 16)
        >>> encoding["_positions/images"].shape
        (2, 1, 2, 2)
        >>> encoding["continuous"].shape
        (2, 1)
        >>> encoding["actions"].shape
        (2, 3)

    Args:
        mu (float, optional): μ parameter for the μ-law companding. Defaults to 100.
        M (float, optional): M parameter for the μ-law companding. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization of continuous values. Defaults to 1024.
        token_shift (int, optional): Shift for the discrete tokens. Defaults to 32_000.
    """

    def __init__(
        self,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        patch_size: int = 16,
        token_shift: int = 32_000,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.patch_size = patch_size
        self.token_shift = token_shift

        self.mu_law_compand = True
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    def tokenize_text(self, x: np.ndarray) -> np.ndarray:
        tokens = self.text_tokenizer(x.tolist())
        return np.array(tokens["input_ids"])

    def extract_patches(self, images: np.ndarray) -> np.ndarray:
        """
        Extract patches from images.

        Args:
            images (np.ndarray): Images to extract patches from of shape (B, C, H, W).

        Returns:
            np.ndarray: Patches extracted from the images. Output has shape (B, N, C, P, P), where P is the patch size.
                Patches are flattened in row-major order.
        """
        B, C, H, W = images.shape
        # First, reshape to the closest above multiple of the patch size
        # cv2 works with channels last, so we need to transpose the image.
        images = images.transpose(0, 2, 3, 1)
        P = self.patch_size
        H = H - H % P + P if H % P != 0 else H
        W = W - W % P + P if W % P != 0 else W
        resized_images = np.zeros((B, H, W, C), dtype=images.dtype)
        for i in range(B):
            resized_images[i] = cv2.resize(images[i], (W, H), interpolation=cv2.INTER_AREA)
        images = resized_images.transpose(0, 3, 1, 2)  # Back to channels first
        patches = images.reshape(B, C, H // P, P, W // P, P).transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, -1, C, P, P)
        # relative position intervals of the patches within the image
        # described with array [[x_min, y_min], [x_max, y_max]]
        # Output shape is (B, N, 2, 2)
        positions = np.array(
            [
                [[i / (H // P), j / (W // P)], [(i + 1) / (H // P), (j + 1) / (W // P)]]
                for i in range(H // P)
                for j in range(W // P)
            ]
        )
        positions = np.tile(positions, (B, 1, 1, 1))

        return patches, positions

    def tokenize_discrete(self, x: np.ndarray) -> np.ndarray:
        # Unsqueeze when the input is a vector
        x = np.array(x, dtype=np.int64)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        tokens = x + self.token_shift
        return tokens

    def tokenize_continuous(self, x: np.ndarray) -> np.ndarray:
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

    def inverse_tokenize_continuous(self, tokens: np.ndarray) -> np.ndarray:
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

    def __call__(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        output = {}
        for key in inputs:
            value = inputs[key]
            if is_text(value):
                output[key] = self.tokenize_text(value)
            elif is_image(value):
                value = np.array(value, dtype=np.uint8)
                if np.argmin(value.shape[1:]) == 2:  # channels last, make it channels first
                    value = np.transpose(value, (0, 3, 1, 2))
                assert np.argmin(value.shape[1:]) == 0, "Channels error"
                output[key], positions = self.extract_patches(value)
                output[f"_positions/{key}"] = positions
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
        "texts": np.array(["Go right", "Go left"]),
        "images": np.random.randint(0, 256, (2, 3, 16, 16), dtype=np.uint8),
        "continuous": np.array([2.1, 3.4]),
        "actions": np.array([[9, 8, 6], [5, 9, 9]]),
    }
    encoding = tokenizer(inputs)
    print(encoding.keys())
