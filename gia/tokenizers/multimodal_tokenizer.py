from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from gia.tokenizers.continuous_tokenizer import ContinuousTokenizer


def is_text(x: Any) -> bool:
    """
    Check if input is text.

    It checks if the input is a string or a list of strings.
    """
    return isinstance(x, str) or isinstance(x, List) and all(isinstance(s, str) for s in x)


def is_image(x: Any) -> bool:
    """
    Check if input is an image.

    Returns True if the input is a torch tensor with 3 or 4 dimensions.
    """
    return isinstance(x, Tensor) and x.dim() in [3, 4]


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
    return isinstance(x, Tensor) and x.dtype == torch.int64


class DiscreteTokenizer(nn.Module):
    """
    Discrete tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = DiscreteTokenizer()
        >>> tensors = torch.tensor([[3, 2, 1, 0],
        ...                         [0, 1, 2, 3],
        ...                         [1, 2, 3, 0]])
        >>> tokenizer(tensors)
        {'input_ids': tensor([[3, 2, 1, 0],
                [0, 1, 2, 3],
                [1, 2, 3, 0]]), 'attention_mask': tensor([[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]])}
    """

    def __init__(self, shift: int = 0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        # Unsqueeze if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        input_ids = x + self.shift
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class MultiModalTokenizer(nn.Module):
    """
    Multi-modal tokenizer.

    Example:
        >>> import torch
        >>> tokenizer = MultiModalTokenizer()
        >>> inputs = {
        ...     "texts": ["Hello world!", "This is a test."],
        ...     "images": torch.rand(2, 3, 84, 84),
        ...     "continuous": torch.rand(2),
        ...     "actions": torch.randint(0, 10, (2, 4)),
        ... }
        >>> encoding = tokenizer(inputs)
        >>> encoding.keys()
        dict_keys(['input_ids', 'attention_mask'])
        >>> encoding["input_ids"]
        tensor([[  101,  7592,  2088,   999,   102,  1012,  1012,  1012,  1012,  1012,


    Args:
        mu (float, optional): Mu-law companding parameter. Defaults to 100.
        M (float, optional): Mu-law companding parameter. Defaults to 256.
        nb_bins (int, optional): Number of bins for the discretization. Defaults to 1024.
    """

    def __init__(
        self,
        mu: float = 100,
        M: float = 256,
        nb_bins: int = 1024,
        token_shift: int = 32_000,
        use_separator: bool = True,
    ):
        super().__init__()
        self.token_shift = token_shift
        self.use_separator = use_separator
        # Token for separating observations and actions
        self.separator_token = torch.tensor(nb_bins + token_shift, dtype=torch.int64)
        self.text_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.image_tokenizer = ...  # TODO:
        self.dicrete_tokenizer = DiscreteTokenizer(token_shift)
        self.continuous_tokenizer = ContinuousTokenizer(mu=mu, M=M, nb_bins=nb_bins, shift=token_shift)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # Get the observation keys
        observation_keys = sorted([key for key in inputs.keys() if key not in ["actions", "dones", "rewards"]])

        tokens = []

        # Observation part
        # Detect the inputs corresponding to texts
        text_keys = [key for key in observation_keys if is_text(inputs[key])]
        for key in sorted(text_keys):
            texts = inputs[key]
            text_tokens = self.text_tokenizer(texts)
            tokens.append(text_tokens)
            # Here, tokens is a list dict, whose keys are input_ids, attention_mask
            # input_ids is a list of list of int
            # attention_mask is a list of list of int (1 for real tokens, 0 for padding)

        # Detect the inputs corresponding to images
        image_keys = [key for key in observation_keys if is_image(inputs)]
        for key in sorted(image_keys):
            images = inputs[key]
            continue
            raise NotImplementedError("Images tokenisation is not implemented yet.")
            image_tokens = self.image_tokenizer(images)
            tokens.append(image_tokens)

        # Detect the inputs corresponding to dicrete tensors
        discrete_tensor_keys = [key for key in observation_keys if is_discrete(inputs[key]) and key != "actions"]
        for key in sorted(discrete_tensor_keys):
            tensors = inputs[key]
            tensor_tokens = self.dicrete_tokenizer(tensors)
            tokens.append(tensor_tokens)

        # Detect the inputs corresponding to continuous tensors
        continuous_tensor_keys = [key for key in observation_keys if is_continuous(inputs[key]) and key != "actions"]
        for key in sorted(continuous_tensor_keys):
            tensors = inputs[key]
            tensor_tokens = self.continuous_tokenizer(tensors)
            tokens.append(tensor_tokens)

        # Add the separator token
        if self.use_separator:
            sequence_length = len(tokens[0]["input_ids"])
            separator_tokens = self.separator_token.repeat(sequence_length, 1)
            tokens.append(
                {
                    "input_ids": separator_tokens,
                    "attention_mask": torch.ones_like(separator_tokens),
                }
            )

        # Action part
        if "actions" in inputs:
            actions = inputs["actions"]
            # Detect the inputs corresponding to dicrete actions
            if actions.dtype == torch.int64:  # If actions are discrete, the tokens are simply the actions
                tokens.append(self.dicrete_tokenizer(actions))
            elif actions.dtype == torch.float32:
                tokens.append(self.continuous_tokenizer(actions))
            else:
                raise ValueError(f"Invalid actions dtype: {actions.dtype}, expected torch.int64 or torch.float32.")

        # Concatenate the tokens
        token_ids = torch.cat([token["input_ids"] for token in tokens], dim=1)
        attention_mask = torch.cat([token["attention_mask"] for token in tokens], dim=1)
        return {"input_ids": token_ids, "attention_mask": attention_mask}
