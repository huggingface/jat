from typing import Dict

import cv2
import numpy as np
from transformers import AutoTokenizer

from gia.config import DatasetArguments
from gia.utils.utils import discretize, inverse_mu_law, mu_law


class MultimodalProcessor:
    """
    Multi-modal tokenizer.

    Args:
        args (:obj:`DatasetArguments`): Dataset arguments.

    Example:
        >>> import numpy as np
        >>> from gia.config import DatasetArguments
        >>> from gia.processor import MultimodalProcessor
        >>> args = DatasetArguments()
        >>> processor = MultimodalProcessor(args)
        >>> inputs = {
        ...     "text_observations": np.array(["Go right", "Go left"]),
        ...     "image_observations": np.random.randint(0, 256, (2, 3, 32, 32), dtype=np.uint8),
        ...     "continuous_actions": np.array([[2.1], [3.4]]),
        ...     "discrete_actions": np.array([[9, 8, 6], [5, 9, 9]]),
        ... }
        >>> encodings = processor(inputs)
        >>> for key, value in encodings.items():
        ...     print(f"{key}: {value.shape}")
        ...
        text_observations: (2, 4)
        text_observations_attention_mask: (2, 4)
        patches_positions: (2, 4, 2, 2)
        image_observations: (2, 4, 3, 16, 16)
        image_observations_attention_mask: (2, 4)
        continuous_actions: (2, 1)
        continuous_actions_attention_mask: (2, 1)
        discrete_actions: (2, 3)
        discrete_actions_attention_mask: (2, 3)
    """

    def __init__(self, args: DatasetArguments) -> None:
        super().__init__()
        self.mu = args.mu
        self.M = args.M
        self.nb_bins = args.nb_bins
        self.patch_size = args.patch_size
        self.token_shift = args.token_shift

        self.mu_law_compand = True
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    def tokenize_text(self, x: np.ndarray) -> np.ndarray:
        output = self.text_tokenizer(x.tolist(), padding="longest")
        tokens = np.array(output["input_ids"])
        attention_mask = np.array(output["attention_mask"], dtype=bool)
        return tokens, attention_mask

    def extract_patches(self, images: np.ndarray) -> np.ndarray:
        """
        Extract patches from images.

        Args:
            images (np.ndarray): Images to extract patches from of shape (B, C, H, W).

        Returns:
            Tuple of:
                - patches (np.ndarray): Patches extracted from the images. Output has shape (B, N, C, P, P), where P
                    is the patch size. Patches are flattened in row-major order.
                - attention_mask (np.ndarray): Attention mask of shape (B, N).
                - patches_positions (np.ndarray): Relative position intervals of the patches. Output has shape
                    (B, N, 2, 2), where the last two dimensions are the start and end positions of the patch.
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
        attention_mask = np.ones(patches.shape[:2], dtype=bool)
        return patches, attention_mask, positions

    def tokenize_discrete(self, x: np.ndarray) -> np.ndarray:
        # Unsqueeze when the input is a vector
        x = np.array(x, dtype=np.int64)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        tokens = x + self.token_shift
        attention_mask = np.ones_like(tokens, dtype=bool)
        return tokens, attention_mask

    def tokenize_continuous(self, x: np.ndarray) -> np.ndarray:
        # Normalize tensors to the range [-1, 1]
        x = np.array(x, dtype=np.float32)
        if self.mu_law_compand:
            x = mu_law(x, mu=self.mu, M=self.M)

        # Clip to the range [-1, 1]
        x = np.clip(x, -1.0, 1.0)

        # Discretize tensors
        x = discretize(x, nb_bins=self.nb_bins)

        # Unsqueeze if needed
        tokens = x + self.token_shift
        attention_mask = np.ones_like(tokens, dtype=bool)
        return tokens, attention_mask

    def inverse_tokenize_continuous(self, tokens: np.ndarray) -> np.ndarray:
        """
        Inverse tokenize continous.

        First, each integer element of input tensor is mapped to the center of the corresponding bin.
        Then, the tensor is de-mu-law companded if needed.

        Args:
            tokens (np.ndarray): Tokens

        Returns:
            Tensor: Reconstructed array
        """
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
                tokens, attention_mask = self.tokenize_text(value)
            elif key.startswith("image"):
                # actually, this is not a token, but patches
                tokens, attention_mask, positions = self.extract_patches(value)
                output["patches_positions"] = positions
            elif key.startswith("discrete"):
                tokens, attention_mask = self.tokenize_discrete(value)
            elif key.startswith("continuous"):
                tokens, attention_mask = self.tokenize_continuous(value)
            else:
                raise ValueError(f"Unknown input type for key '{key}'.")

            output[key] = tokens
            output[f"{key}_attention_mask"] = attention_mask
        return output
