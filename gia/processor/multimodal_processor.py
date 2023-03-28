from typing import Dict

import cv2
import numpy as np
from transformers import AutoTokenizer

from gia.utils.utils import discretize, inverse_mu_law, mu_law


class MultimodalProcessor:
    """
    Multi-modal tokenizer.

    Example:
        >>> import numpy as np
        >>> tokenizer = MultimodalProcessor()
        >>> inputs = {
        ...     "text_observations": np.array(["Go right", "Go left"]),
        ...     "image_observations": np.random.randint(0, 256, (2, 3, 32, 32), dtype=np.uint8),
        ...     "continuous_actions": np.array([2.1, 3.4]),
        ...     "discrete_actions": np.array([[9, 8, 6], [5, 9, 9]]),
        ... }
        >>> encoding = tokenizer(inputs)
        >>> for key in encoding:
        ...     print(f"{key}: {encoding[key].shape}")
        text_observations: (2, 4)
        patches_positions: (2, 4, 2, 2)
        image_observations: (2, 4, 3, 16, 16)
        continuous_actions: (2, 1)
        discrete_actions: (2, 3)

    Args:
        mu (float, optional): μ parameter for the μ-law companding. Defaults to 100.
        M (float, optional): M parameter for the μ-law companding. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization of continuous values. Defaults to 1024.
        patch_size (int, optional): Size of the patches to extract from the images. Defaults to 16.
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
            ],
            dtype=np.float32,
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
            tokens (np.ndarray): Tokens of shape (..., 1)

        Returns:
            Tensor: Reconstructed array
        """
        # The tokens array for continuous values are expected to have shape (..., 1), so we squeeze it
        tokens = np.squeeze(tokens, axis=-1)

        # Subtract token shift
        tokens = tokens - self.token_shift

        # Maps tokens from [0, nb_bins-1] to [-1, 1]
        # We map the bin number to the center of the bin
        x = (2 * tokens + 1) / self.nb_bins - 1

        # De-mu-law compand tensors
        if self.mu_law_compand:
            x = inverse_mu_law(x, mu=self.mu, M=self.M)

        return x

    def __call__(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        output = {}
        for key in inputs.keys():
            value = inputs[key]
            if key.startswith("text"):
                tokens = self.tokenize_text(value)
            elif key.startswith("image"):
                tokens, positions = self.extract_patches(value)  # actually, this is not a token, but patches
                output["patches_positions"] = positions
            elif key.startswith("discrete"):
                tokens = self.tokenize_discrete(value)
            elif key.startswith("continuous"):
                tokens = self.tokenize_continuous(value)
            else:
                raise ValueError(f"Unknown input type for key '{key}'.")

            output[key] = tokens
        return output
