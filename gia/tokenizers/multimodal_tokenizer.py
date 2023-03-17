from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from gia.tokenizers.continuous_tokenizer import ContinuousTokenizer


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


def transform_func(examples):
    """Convert image to uint8.

    Fix of https://github.com/huggingface/datasets/issues/4623
    """
    for key in examples.keys():
        if isinstance(examples, Tensor) and examples[key].dim() == 4:
            examples[key] = examples[key].to(dtype=torch.uint8)
    return examples


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
        use_separator (bool, optional): Whether to use a separator token between observations and actions. Defaults to True.
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
        self.boolean_tokenizer = BooleanTokenizer(token_shift)
        self.continuous_tokenizer = ContinuousTokenizer(mu=mu, M=M, nb_bins=nb_bins, shift=token_shift)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        inputs = transform_func(inputs)

        output = {}
        for key, value in inputs.items():
            if is_text(value):
                tokens = self.text_tokenizer(value)
            elif is_image(value):
                continue
                tokens = self.image_tokenizer(value)
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
            # Here, tokens is a list dict, whose keys are tokens, attention_mask
            # tokens is a list of list of int
            # attention_mask is a list of list of int (1 for real tokens, 0 for padding)

        # Detect the inputs corresponding to images
        image_keys = [key for key in observation_keys if is_image(inputs[key])]
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
            sequence_length = len(tokens[0]["tokens"])
            separator_tokens = self.separator_token.repeat(sequence_length, 1)
            tokens.append(
                {
                    "tokens": separator_tokens.tolist(),
                    "attention_mask": torch.ones_like(separator_tokens).tolist(),
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
        tokens = concat_multi_dim_iterables([token["tokens"] for token in tokens])
        attention_mask = concat_multi_dim_iterables([token["attention_mask"] for token in tokens])
        loss_mask = concat_multi_dim_iterables([token["loss_mask"] for token in tokens])
        return {"tokens": tokens, "attention_mask": attention_mask, "loss_mask": loss_mask}
