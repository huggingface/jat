import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GEGLU(nn.Module):
    r"""Applies the Gaussian Error Gated Linear Units function (GEGLU) as described in
    'GLU Variants Improve Transformer' (Shazeer, 2020):

    Given the input tensor x, the function splits x into two tensors 'a' and 'b' along the last dimension. Then it
    applies the GELU function to 'b' and multiplies the result by 'a':

    .. math:: \text{GEGLU}(x) = a * \text{GELU}(b)

    where a and b are two equally-sized chunks obtained by splitting the input tensor x.

    Args:
        x (Tensor): The input tensor. The size of its last dimension must be even, as it is split into two equal
            chunks.

    Shape:
        - Input: :math:`(*, H)` where :math:`*` represents any number of dimensions and `H` is the dimension size that
            will be split.
        - Output: :math:`(*, H/2)`, where the output size is half the size of the input's last dimension.

    Examples::

        >>> m = nn.GEGLU()
        >>> input = torch.randn(2, 4)
        >>> output = m(input)
    """

    def geglu(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Input tensor must have an even number of dimensions along the last axis. "
                f"Got {x.shape[-1]} dimensions."
            )
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.geglu(x)
