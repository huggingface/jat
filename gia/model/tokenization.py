from typing import List, Optional

import torch
from torch import Tensor, nn

from gia.utils.utils import discretize, inverse_mu_law, mu_law


def tokenize_continuous(x: Tensor, mu_law_compand: bool = True, mu: float = 100, M: float = 256, nb_bins: int = 1024):
    """
    Tokenize continous.

    First, each floating point element of input tensor is mu-law companded as in WaveNet (Oord et al., 2016).
    Then, the tensor is discretized into integers in the range [0, nb_bins-1].

    Args:
        x (Tensor): Input tensor
        mu_law_encoding (bool, optional): Whether to use mu-law compading. Defaults to True.
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.

    Returns:
        Tensor: Tokens
    """
    # Normalize tensors
    if mu_law_compand:
        x = mu_law(x, mu=mu, M=M)

    # Clip to the range [-1, 1]
    x = torch.clamp(x, -1.0, 1.0)

    # Discretize tensors
    return discretize(x, nb_bins=nb_bins)


def inverse_tokenize_continuous(
    tokens: Tensor, mu_law_companded: bool = True, mu: float = 100, M: float = 256, nb_bins: int = 1024
):
    """
    Inverse tokenize continous.

    First, each integer element of input tensor is mapped to the center of the corresponding bin.
    Then, the tensor is de-mu-law companded if needed.

    Args:
        tokens (Tensor): Tokens
        mu_law_companded (bool, optional): Whether inputs are mu-law companded. Defaults to True.
        mu (float, optional): μ parameter. Defaults to 100.
        M (float, optional): M parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.

    Returns:
        Tensor: Reconstructed tensor
    """
    # Maps tokens from [0, nb_bins-1] to [-1, 1]
    # We map the bin number to the center of the bin
    x = (2 * tokens + 1) / nb_bins - 1

    # De-mu-law compand tensors
    if mu_law_companded:
        x = inverse_mu_law(x, mu=mu, M=M)

    return x


class Tokenizer(nn.Module):
    """
    Tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = Tokenizer()
        >>> tensors = torch.tensor([[-0.6285, -5.1349,  3.9317, -3.5854],
        ...                         [ 3.9263, -1.8975,  7.3103, -7.3467],
        ...                         [-6.9724, -2.9982,  9.1740,  0.1002]])
        >>> actions = torch.tensor([[ 3.4870,  0.7328],
        ...                         [-3.1476,  3.7754],
        ...                         [ 1.8333, -1.5895]])
        >>> tokenizer(tensors=tensors, actions=actions)
        tensor([[32302, 32197, 32813, 32215, 33024, 32807, 32729],
                [32813, 32247, 32844, 32179, 33024, 32221, 32811],
                [32181, 32224, 32856, 32633, 33024, 32775, 32256]])

    Args:
        mu (float, optional): Mu-law companding parameter. Defaults to 100.
        M (float, optional): Mu-law companding parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.
    """

    def __init__(self, mu: float = 100, M: float = 256, nb_bins: int = 1024, token_shift: int = 32_000):
        super().__init__()
        self.mu = mu
        self.M = M
        self.nb_bins = nb_bins
        self.token_shift = token_shift
        # Token for separating observations and actions
        self.separator_token = torch.tensor(self.nb_bins + token_shift, dtype=torch.long)

    def forward(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[Tensor] = None,
        tensors: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        compand_tensors: bool = True,
        compand_actions: bool = True,
    ) -> Tensor:
        # Get the sequence length
        for x in [texts, images, tensors, actions]:
            if x is not None:
                sequence_length = len(x)
                break
        else:
            raise RuntimeError("At least one of the inputs must be provided.")

        # Tokenize observations and actions and concatenate them following the order:
        # [text_tokens, image_tokens, tensor_tokens, separator_token, action_tokens]
        tokens = []
        if texts is not None:
            raise NotImplementedError("Texts are not implemented yet.")
            text_tokens = ...
            tokens.append(text_tokens)
        if images is not None:
            raise NotImplementedError("Images are not implemented yet.")
            image_tokens = ...
            tokens.append(image_tokens)
        if tensors is not None:
            tensor_tokens = tokenize_continuous(tensors, mu_law_compand=compand_tensors)
            tensor_tokens = tensor_tokens + self.token_shift
            tokens.append(tensor_tokens)
        separator_tokens = self.separator_token.repeat(sequence_length, 1)
        tokens.append(separator_tokens)
        if actions is not None:
            action_tokens = tokenize_continuous(actions, mu_law_compand=compand_actions)
            action_tokens = action_tokens + self.token_shift
            tokens.append(action_tokens)
        return torch.concat(tokens, dim=1)
