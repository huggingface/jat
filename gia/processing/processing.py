from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from transformers import AutoTokenizer

from gia.config import DatasetArguments
from gia.utils.utils import discretize, inverse_mu_law, mu_law


class GiaProcessor:
    """
    Processor for the Gia model.

    Args:
        args (:obj:`DatasetArguments`): Dataset arguments.

    Example:
        >>> import numpy as np
        >>> from gia.config import DatasetArguments
        >>> from gia.processing import GiaProcessor
        >>> args = DatasetArguments()
        >>> processor = GiaProcessor(args)
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
        self.seq_len = args.seq_len

        self.mu_law_compand = True
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.token_shift = self.text_tokenizer.vocab_size
        self.vocab_size = self.token_shift + self.nb_bins

    def tokenize_text(self, x: List[str]) -> np.ndarray:
        output = self.text_tokenizer(x)
        tokens = np.array(output["input_ids"])
        return tokens

    def extract_patches(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        # Unsqueeze if needed
        tokens = x + self.token_shift
        return tokens

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

    def __call__(
        self,
        continuous_observations: Optional[List[Optional[np.ndarray]]] = None,
        discrete_observations: Optional[List[Optional[np.ndarray]]] = None,
        text_observations: Optional[List[Optional[List[str]]]] = None,
        image_observations: Optional[List[Optional[np.ndarray]]] = None,
        continuous_actions: Optional[List[Optional[np.ndarray]]] = None,
        discrete_actions: Optional[List[Optional[np.ndarray]]] = None,
        rewards: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[str, np.ndarray]:
        # Example:
        # continuous_observations = [[a1, a2, ...], None],
        # discrete_observations = [None, [c1, c2, ...]],
        # continuous_actions = [[b1, b2, ...], [d1, d2, ...]],
        # discrete_actions = [None, None],
        # text_observations = [["hello", "world", ...], None],

        # Desired output (t = tokenized):
        # output = {
        #     "continuous_observations": [[ta1, ta2, ...], None],
        #     "discrete_observations": [None, [tc1, tc2, ...]],
        #     "continuous_actions": [[tb1, tb2, ...], [td1, td2, ...]],
        #     "discrete_actions": [None, None],
        #     "text_observations": [[[tt2, tt2], [tt3, tt4, tt5], ...], None],
        # }
        output = {}
        if continuous_observations is not None:  # This modality appears at least once in the batch
            output["continuous_observations"] = []
            for observations in continuous_observations:
                if observations is not None:
                    # observations = [o1, o2, ...] is actually the episode
                    tokens = self.tokenize_continuous(observations)
                    output["continuous_observations"].append(tokens)
                else:  # There is no such modality in this episode
                    output["continuous_observations"].append(None)

        if discrete_observations is not None:  # This modality appears at least once in the batch
            output["discrete_observations"] = []
            for observations in discrete_observations:
                if observations is not None:
                    # observations = [o1, o2, ...] is actually the episode
                    tokens = self.tokenize_discrete(observations)
                    output["discrete_observations"].append(tokens)
                else:  # There is no such modality in this episode
                    output["discrete_observations"].append(None)

        if text_observations is not None:  # This modality appears at least once in the batch
            output["text_observations"] = []
            for observations in text_observations:
                if observations is not None:
                    # observations = ["hello", "world", ...] is actually the episode
                    tokens = self.tokenize_text(observations)
                    output["text_observations"].append(tokens)
                else:  # There is no such modality in this episode
                    output["text_observations"].append(None)

        if image_observations is not None:  # This modality appears at least once in the batch
            output["image_observations"] = []
            output["image_observations_patches_positions"] = []
            for observations in image_observations:
                if observations is not None:
                    # observations = ["hello", "world", ...] is actually the episode
                    patches, positions = self.extract_patches(observations)
                    output["image_observations"].append(patches)
                    output["image_observations_patches_positions"] = positions
                else:  # There is no such modality in this episode
                    output["image_observations"].append(None)
                    output["image_observations_patches_positions"].append(None)

        if continuous_actions is not None:  # This modality appears at least once in the batch
            output["continuous_actions"] = []
            for actions in continuous_actions:
                if actions is not None:
                    # actions = [a1, a2, ...] is actually the episode
                    tokens = self.tokenize_continuous(actions)
                    output["continuous_actions"].append(tokens)
                else:  # There is no such modality in this episode
                    output["continuous_actions"].append(None)

        if discrete_actions is not None:  # This modality appears at least once in the batch
            output["discrete_actions"] = []
            for actions in discrete_actions:
                if actions is not None:
                    # actions = [a1, a2, ...] is actually the episode
                    tokens = self.tokenize_discrete(actions)
                    output["discrete_actions"].append(tokens)
                else:
                    output["discrete_actions"].append(None)

        if rewards is not None:  # The reward appears at least once in the batch
            output["rewards"] = rewards  # Currently, we do not tokenize the rewards

        return output
