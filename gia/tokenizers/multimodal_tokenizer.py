from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from gia.tokenizers.continuous_tokenizer import ContinuousTokenizer


def is_text(x: Any):
    return isinstance(x, str)


def is_image(x: Any):
    return isinstance(x, torch.Tensor) and x.ndim == 3


def concat_dicts(dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return


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
        input_ids = x.unsqueeze(1) + self.shift
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
        }
        >>> tokenizer(inputs)

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
        tokens = []
        # Remove dones and rewards from the keys
        keys = sorted([key for key in inputs.keys() if key not in ["dones", "rewards"]])

        # Observation part
        # Detect the inputs corresponding to texts
        text_keys = [key for key in keys if is_text(inputs[key][0])]
        for key in sorted(text_keys):
            texts = inputs[key]
            text_tokens = self.text_tokenizer(texts)
            # convert to tensors # TODO: is there a better way?
            text_tokens = {key: torch.tensor(text_tokens[key]) for key in ["input_ids", "attention_mask"]}
            tokens.append(text_tokens)
            # Here, tokens is a list dict, whose keys are input_ids, attention_mask
            # input_ids is a list of list of int
            # attention_mask is a list of list of int (1 for real tokens, 0 for padding)

        # Detect the inputs corresponding to images
        image_keys = [key for key in keys if is_image(inputs[key][0])]
        for key in sorted(image_keys):
            images = inputs[key]
            continue
            raise NotImplementedError("Images tokenisation is not implemented yet.")
            image_tokens = self.image_tokenizer(images)
            tokens.append(image_tokens)

        # Detect the inputs corresponding to dicrete tensors
        discrete_tensor_keys = [key for key in keys if inputs[key].dtype == torch.int64 and key != "actions"]
        for key in sorted(discrete_tensor_keys):
            tensors = inputs[key]
            tensor_tokens = self.dicrete_tokenizer(tensors)
            tokens.append(tensor_tokens)

        # Detect the inputs corresponding to continuous tensors
        continuous_tensor_keys = [key for key in keys if inputs[key].dtype == torch.float32 and key != "actions"]
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
            actions = inputs.pop("actions")
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
