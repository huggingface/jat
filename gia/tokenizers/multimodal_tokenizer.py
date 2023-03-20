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


def is_bool(x: Any) -> bool:
    """
    Check if input is boolean.

    Returns True if the input is a torch tensor with dtype torch.bool.
    """
    return isinstance(x, Tensor) and x.dtype == torch.bool and x.dim() == 1


def concat_multi_dim_iterables(l: List[List[Iterable]]) -> List[List]:
    """
    Concatenate a list of iterables from shape (M, N, X) to shape (N, X1 + X2 + ...).

    Examples:
        >>> concat_multi_dim_iterables([
        ... [[1, 2], [3, 4, 5]],
        ... [[6, 7, 8], [9]],
        ... ])
        [[1, 2, 6, 7, 8], [3, 4, 5, 9]]

    Args:
        l (List[List[Iterable]]): A list of iterables of shape (M, N, X) where X can vary for each element.

    Returns:
        List[List]: A list of iterables of shape (N, X1 + X2 + ...).
    """
    output = []
    for idx in range(len(l[0])):
        sub_list = []
        for iterable in l:
            sub_list.extend(iterable[idx])
        output.append(sub_list)
    return output


def transform_func(examples):
    """Convert image to uint8.

    Fix of https://github.com/huggingface/datasets/issues/4623
    """
    for key in examples.keys():
        if isinstance(examples, Tensor) and examples[key].dim() == 4:
            examples[key] = examples[key].to(dtype=torch.uint8)
    return examples


class DiscreteTokenizer(nn.Module):
    """
    Discrete tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = DiscreteTokenizer()
        >>> tensors = torch.tensor([[3, 2, 1],
        ...                         [0, 1, 2]])
        >>> tokenizer(tensors)
        {'tokens': [[3, 2, 1], [0, 1, 2]], 'attention_mask': [[1, 1, 1], [1, 1, 1]]}
    """

    def __init__(self, shift: int = 0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        # Unsqueeze when the input is a vector
        if x.dim() == 1:
            x = x.unsqueeze(1)
        tokens = x + self.shift
        return {
            "tokens": tokens.tolist(),
            "attention_mask": torch.ones_like(tokens).tolist(),
        }


class BooleanTokenizer(DiscreteTokenizer):
    """
    Boolean tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = BooleanTokenizer()
        >>> tensors = torch.tensor([[ True, False, True],
        ...                         [False,  True, True]])
        >>> tokenizer(tensors)
        {'tokens': [[1, 0, 1], [0, 1, 1]], 'attention_mask': [[1, 1, 1], [1, 1, 1]]}
    """

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(torch.int64)
        return super().forward(x)


class ContinuousTokenizer(nn.Module):
    """
    Continous tokenizer.

    First, each floating point element of input tensor is mu-law companded as in WaveNet (Oord et al., 2016).
    Then, the tensor is discretized into integers in the range [0, nb_bins-1].

    Example:
        >>> import torch
        >>> tokenizer = ContinuousTokenizer()
        >>> tensors = torch.tensor([[-0.6285, -5.1349,  3.9317, -3.5854],
        ...                         [ 3.9263, -1.8975,  7.3103, -7.3467],
        ...                         [-6.9724, -2.9982,  9.1740,  0.1002]])
        >>> tokenizer(tensors)
        tensor([[302, 197, 813, 215],
                [813, 247, 844, 179],
                [181, 224, 856, 633]])

    Args:
        mu_law_compand (bool, optional): Whether to use mu-law companding. Defaults to True.
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.
    """

    def __init__(
        self, mu_law_compand: bool = True, mu: float = 100, M: float = 256, nb_bins: int = 1024, shift: int = 0
    ) -> None:
        super().__init__()
        self.mu_law_compand = mu_law_compand
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.shift = shift

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
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
        tokens = x + self.shift
        return {
            "tokens": tokens.tolist(),
            "attention_mask": torch.ones_like(tokens).tolist(),
        }

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


class PatchExtractor(nn.Module):
    """
    Extract patches from images.

    Args:
        patch_size (int, optional): Patch size. Defaults to 16.

    Returns:
        dict: A dictionary containing the following keys and values:
            - "patches": A tensor of shape (batch_size, H*W, channels, patch_size, patch_size).
            - "row_indices": A tensor of shape (batch_size, H*W) containing row indices of patches.
            - "col_indices": A tensor of shape (batch_size, H*W) containing column indices of patches.
            - "attention_mask": A tensor of shape (batch_size, H*W) containing attention mask.

    Examples:
        >>> x = torch.randn(2, 3, 16, 12)
        >>> patch_extractor = PatchExtractor(patch_size=4)
        >>> result = patch_extractor(x)
        >>> result["patches"].shape
        torch.Size([2, 12, 3, 4, 4])
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

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        assert height % self.patch_size == 0, "Height must be divisible by patch_size"
        assert width % self.patch_size == 0, "Width must be divisible by patch_size"
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Get H and W
        batch_size, H, W = patches.shape[:3]
        # Get the row and the column indices. First dim is still the batch size
        rows, cols = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        row_indices = rows.flatten().unsqueeze(0).repeat(batch_size, 1)
        col_indices = cols.flatten().unsqueeze(0).repeat(batch_size, 1)
        # Flatten the patches (from shape (batch_size, H, W, channels, patch_size, patch_size)
        # to (batch_size, H*W, channels, patch_size, patch_size))
        patches = patches.flatten(1, 2)
        return {
            "patches": patches,
            "row_indices": row_indices.tolist(),
            "col_indices": col_indices.tolist(),
            "attention_mask": torch.ones(batch_size, H * W, dtype=torch.int).tolist(),
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
        # Token for separating observations and actions
        self.separator_token = torch.tensor(nb_bins + token_shift, dtype=torch.int64)
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.patch_extractor = PatchExtractor(patch_size)
        self.dicrete_tokenizer = DiscreteTokenizer(token_shift)
        self.boolean_tokenizer = BooleanTokenizer(token_shift)
        self.continuous_tokenizer = ContinuousTokenizer(mu=mu, M=M, nb_bins=nb_bins, shift=token_shift)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        inputs = transform_func(inputs)

        output = {}
        for key, value in inputs.items():
            if is_text(value):
                tokens = self.text_tokenizer(value)
                # Rename input_ids to tokens
                tokens["tokens"] = tokens.pop("input_ids")
            elif is_image(value):  # Warning: special case for images: output is a patch, not a number
                tokens = self.patch_extractor(value)
            elif is_discrete(value):
                tokens = self.dicrete_tokenizer(value)
            elif is_continuous(value):
                tokens = self.continuous_tokenizer(value)
            elif is_bool(value):
                tokens = self.boolean_tokenizer(value)
            else:
                raise ValueError(f"Unknown input type for key '{key}'.")

            for _key, _value in tokens.items():
                output[f"{key}_{_key}"] = _value
        return output


if __name__ == "__main__":
    image = torch.rand(2, 3, 16, 12)
    text = ["Go right", "Go left"]
    multimodal_tokenizer = MultiModalTokenizer(patch_size=4)
    encoding = multimodal_tokenizer({"images": image, "texts": text})
    print(encoding)
