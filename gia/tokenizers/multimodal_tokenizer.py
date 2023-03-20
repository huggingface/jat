from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from gia.utils.utils import discretize, inverse_mu_law, mu_law


def is_text(x: Any) -> bool:
    """
    Check if input is text.

    It checks if the input a list of strings.
    """
    return isinstance(x, List) and all(isinstance(s, str) for s in x)


def is_image(x: Any) -> bool:
    """
    Check if input is an image.

    Returns True if the input is a torch tensor with 4 dimensions.
    """
    return isinstance(x, Tensor) and x.dim() == 4  # shape (batch_size, channels, height, width)


def is_continuous(x: Any) -> bool:
    """
    Check if input is continuous.

    Returns True if the input is a torch tensor with dtype torch.float32.
    """
    return isinstance(x, Tensor) and x.dtype == torch.float32


def is_discrete(x: Any) -> bool:
    """
    Check if input is discrete.

    Returns True if the input is a torch tensor with dtype torch.int64.
    """
    return isinstance(x, Tensor) and x.dtype == torch.int64 and x.dim() == 1


def transform_func(examples):
    """Convert image to uint8.

    Fix of https://github.com/huggingface/datasets/issues/4623
    """
    for key in examples.keys():
        if isinstance(examples, Tensor) and examples[key].dim() == 4:
            examples[key] = examples[key].to(dtype=torch.uint8)
    return examples


class PositionExtractor(nn.Module):
    """
    Extract patches from images.

    Args:
        patch_size (int, optional): Patch size. Defaults to 16.

    Returns:
        dict: A dictionary containing the following keys and values:
            - "row_indices": A tensor of shape (batch_size, H*W) containing row indices of patches.
            - "col_indices": A tensor of shape (batch_size, H*W) containing column indices of patches.
            - "attention_mask": A tensor of shape (batch_size, H*W) containing attention mask.

    Examples:
        >>> x = torch.randn(2, 3, 16, 12)
        >>> position_extractor = PatchExtractor(patch_size=4)
        >>> result = position_extractor(x)
        >>> result["row_indices"]
        [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]]
        >>> result["col_indices"]
        [[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]
        >>> result["attention_mask"]
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    """

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self._cnn = nn.Conv2d(3, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        h_out = height // self.patch_size
        w_out = width // self.patch_size

        # Get the row and the column indices. First dim is still the batch size
        rows, cols = torch.meshgrid(torch.arange(h_out), torch.arange(w_out), indexing="ij")
        row_indices = rows.flatten().unsqueeze(0).repeat(batch_size, 1)
        col_indices = cols.flatten().unsqueeze(0).repeat(batch_size, 1)
        return {
            "row_indices": row_indices.tolist(),
            "col_indices": col_indices.tolist(),
            "attention_mask": torch.ones(batch_size, h_out * w_out, dtype=torch.int).tolist(),
        }


class MultiModalTokenizer(nn.Module):
    """
    Multi-modal tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = MultiModalTokenizer()
        >>> inputs = {
        ...     "texts": ["Go right", "Go left"],
        ...     # "images": torch.rand(2, 3, 224, 224),
        ...     "continuous": torch.tensor([2.1, 3.4]),
        ...     "actions": torch.tensor([[9, 8, 6], [5, 9, 9]]),
        ... }
        >>> encoding = tokenizer(inputs)
        >>> encoding.keys()
        dict_keys(['tokens', 'attention_mask'])
        >>> encoding["tokens"]
        [[2, 162, 193, 3, 32781, 33024, 32009, 32008, 32006], [2, 162, 225, 3, 32806, 33024, 32005, 32009, 32009]]
        >>> encoding["attention_mask"]
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]

    Args:
        mu (float, optional): Mu-law companding parameter. Defaults to 100.
        M (float, optional): Mu-law companding parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.
        token_shift (int, optional): Shift for the discrete tokens. Defaults to 32_000.
        patch_size (int, optional): Patch size. Defaults to 16.
        use_separator (bool, optional): Whether to use a separator token between observations and actions. Defaults to True.
    """

    def __init__(
        self,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        token_shift: int = 32_000,
        patch_size: int = 16,
        use_separator: bool = True,
    ) -> None:
        super().__init__()
        self.token_shift = token_shift
        self.use_separator = use_separator
        self.mu_law_compand = True
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.token_shift = token_shift
        # Token for separating observations and actions
        self.separator_token = torch.tensor(nb_bins + token_shift, dtype=torch.int64)
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    def tokenize_discrete(self, x: Tensor) -> Tensor:
        # Unsqueeze when the input is a vector
        if x.dim() == 1:
            x = x.unsqueeze(1)
        tokens = x + self.token_shift
        return tokens.tolist()

    def tokenize_continuous(self, x: Tensor) -> Tensor:
        # Normalize tensors to the range [-1, 1]
        if self.mu_law_compand:
            x = mu_law(x, mu=self.mu, M=self.M)

        # Clip to the range [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)

        # Discretize tensors
        x = discretize(x, nb_bins=self.nb_bins)

        # Unsqueeze when the input is a vector
        if x.dim() == 1:
            x = x.unsqueeze(1)

        # Unsqueeze if needed
        tokens = x + self.token_shift
        return tokens.tolist()

    def inverse_tokenize_continuous(self, tokens: Tensor) -> Tensor:
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

    def tokenize_text(self, x: Tensor) -> Tensor:
        x[0] = "Hi"
        tokens = self.text_tokenizer(x, padding=True)
        return tokens["input_ids"]

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        inputs = transform_func(inputs)
        # Tokenize just the observations and actions
        # Allow for multiple observations and actions (e.g. key = "observations_0" or "observations/image")
        keys = [key for key in inputs.keys() if key.startswith("observations") or key.startswith("actions")]

        output = {}
        for key in keys:
            value = inputs[key]
            if is_text(value):
                output[f"{key}_tokens"] = self.tokenize_text(value)
            elif is_image(value):  # Warning: special case for images: we do nothing
                continue
            elif is_discrete(value):
                output[f"{key}_tokens"] = self.tokenize_discrete(value)
            elif is_continuous(value):
                output[f"{key}_tokens"] = self.tokenize_discrete(value)
            else:
                raise ValueError(f"Unknown input type for key '{key}'.")
        return output


if __name__ == "__main__":
    image = torch.rand(2, 3, 16, 12)
    text = ["Go right", "Go left"]
    multimodal_tokenizer = MultiModalTokenizer(patch_size=4)
    encoding = multimodal_tokenizer({"images": image, "texts": text})
    print(encoding)
