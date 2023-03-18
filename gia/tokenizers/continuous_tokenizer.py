from typing import Dict

import torch
from torch import Tensor, nn

from gia.utils.utils import discretize, inverse_mu_law, mu_law


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
