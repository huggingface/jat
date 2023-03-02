from typing import List, Optional

import torch
from torch import Tensor, nn

from gia.tokenizers.continuous_tokenizer import ContinuousTokenizer


class MultiModalTokenizer(nn.Module):
    """
    Multi-modal tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = MultiModalTokenizer()
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
        self.token_shift = token_shift
        # Token for separating observations and actions
        self.separator_token = torch.tensor(nb_bins + token_shift, dtype=torch.int64)
        self.text_tokenizer = ...  # TODO:
        self.image_tokenizer = ...  # TODO:
        self.continuous_tokenizer = ContinuousTokenizer(mu=mu, M=M, nb_bins=nb_bins)

    def forward(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[Tensor] = None,
        tensors: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
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
            raise NotImplementedError("Texts tokenisation is not implemented yet.")
            text_tokens = self.text_tokenizer(texts)
            tokens.append(text_tokens)

        if images is not None:
            raise NotImplementedError("Images tokenisation is not implemented yet.")
            image_tokens = self.image_tokenizer(iamges)
            tokens.append(image_tokens)

        if tensors is not None:
            if tensors.dtype == torch.int64:  # If tensors are discrete, the tokens are simply the tensors
                tensor_tokens = tensors
            elif tensors.dtype == torch.float32:
                tensor_tokens = self.continuous_tokenizer(tensors)
            else:
                raise ValueError(f"Invalid tensors dtype: {tensors.dtype}, expected torch.int64 or torch.float32.")
            tensor_tokens = tensor_tokens + self.token_shift
            tokens.append(tensor_tokens)

        separator_tokens = self.separator_token.repeat(sequence_length, 1)
        tokens.append(separator_tokens)

        if actions is not None:
            if actions.dtype == torch.int64:  # If actions are discrete, the tokens are simply the actions
                action_tokens = actions
            elif actions.dtype == torch.float32:
                action_tokens = self.continuous_tokenizer(actions)
            else:
                raise ValueError(f"Invalid actions dtype: {actions.dtype}, expected torch.int64 or torch.float32.")
            action_tokens = action_tokens + self.token_shift
            tokens.append(action_tokens)

        return torch.concat(tokens, dim=1)
