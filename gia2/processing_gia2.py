from typing import Any, Dict, List, Optional, Union

import torch
from torchvision.transforms.functional import to_tensor
from transformers import BatchEncoding, image_utils
from transformers.processing_utils import ProcessorMixin


def truncate(encoding: Dict[str, List[List[Any]]], max_len: int, preserve: bool = False) -> Dict[str, List[List[Any]]]:
    """
    Truncate the sequences in the encoding to the specified maximum length.

    This function is designed to process batch of sequences represented in the encoding dictionary.
    Depending on the chosen strategy, sequences are either truncated with loss of residual data or with preservation
    and incorporation of residual data into the batch.

    Args:
        encoding (`Mapping`):
            A dictionary where each key-value pair consists of a feature name and its corresponding batch of sequences.
            The sequences are expected to be lists.
        max_len (`int`):
            The maximum allowable length for the sequences.
        preserve (`bool`, **optional**):
            Whether to preserve the residual data by adding them as new sequences in the batch. Defaults to `False`.

    Returns:
        `Dict[str, List[List[Any]]]`:
            A dictionary with the same keys as the input `encoding`, containing the truncated batch of sequences.
            If `preserve` is set to `True`, the batch size may increase due to the addition of new sequences formed
            from the residual data.

    Example:

        >>> encoding = {'feature1': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]}
        >>> truncate(encoding, 3, preserve=False)
        {'feature1': [[1, 2, 3], [6, 7, 8]]}

        >>> truncate(encoding, 3, preserve=True)
        {'feature1': [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10]]}
    """
    truncated_encoding = {}

    for key, sequences in encoding.items():
        if not all(isinstance(seq, list) for seq in sequences):
            raise TypeError(f"All sequences under key {key} should be of type list.")

        truncated_sequences = []

        for seq in sequences:
            if len(seq) <= max_len:
                truncated_sequences.append(seq)
                continue

            if preserve:  # truncate and append the residual as new sequences
                truncated_sequences.extend([seq[i : i + max_len] for i in range(0, len(seq), max_len)])
            else:  # simply truncate the sequence
                truncated_sequences.append(seq[:max_len])

        truncated_encoding[key] = truncated_sequences

    return truncated_encoding


def pad(encoding: Dict[str, List[List[Any]]], target_length: int) -> Dict[str, List[List[Any]]]:
    """
    Pad the sequences in the encoding to the specified maximum length.

    This function is designed to process batch of sequences represented in the encoding dictionary.
    The padding value is set to be the first element in the sequence.

    Args:
        encoding (`Mapping`):
            A dictionary where each key-value pair consists of a feature name and its corresponding batch of sequences.
            The sequences are expected to be lists.
        target_length (`int`):
            The desired length for the sequences.

    Returns:
        `Dict[str, List[List[Any]]]`:
            A dictionary with the same keys as the input `encoding`, containing the padded batch of sequences.
            An additional key `attention_mask` is added to the dictionary to indicate the positions of the non-padding
            elements with 1s and the padding elements with 0s. If the input `encoding` already contains an
            `attention_mask` key, the corresponding mask will be updated such that the original masking is preserved,
            and the newly added padding elements will be masked with 0s. In other words, the resulting
            `attention_mask` is a logical "AND" between the provided mask and the mask created due to padding, ensuring
            that any element masked originally remains masked.

    Example:

        >>> encoding = {'feature1': [[1, 2], [3, 4, 5]]}
        >>> pad(encoding, 4)
        {'feature1': [[1, 2, 1, 1], [3, 4, 5, 3]], 'attention_mask': [[1, 1, 0, 0], [1, 1, 1, 0]]}

        >>> encoding = {'feature1': [[1, 2], [3, 4, 5]], "attention_mask": [[1, 0], [0, 1, 1]]}
        >>> pad(encoding, 4)
        {'feature1': [[1, 2, 1, 1], [3, 4, 5, 3]], 'attention_mask': [[1, 0, 0, 0], [0, 1, 1, 0]]}
    """
    padded_encoding = {}

    for key, sequences in encoding.items():
        if not all(isinstance(seq, list) for seq in sequences):
            raise TypeError(f"All sequences under key {key} should be of type list.")
        if key == "attention_mask":  # attention_mask is handled separately
            continue

        padded_sequences = []
        pad_mask = []

        for seq in sequences:
            pad_len = target_length - len(seq)
            padded_seq = seq + [seq[0]] * max(0, pad_len)
            mask = [1] * len(seq) + [0] * max(0, pad_len)

            padded_sequences.append(padded_seq)
            pad_mask.append(mask)

        padded_encoding[key] = padded_sequences

    if "attention_mask" in encoding:
        padded_encoding["attention_mask"] = [
            [a * (b[i] if i < len(b) else 0) for i, a in enumerate(row)]
            for row, b in zip(pad_mask, encoding["attention_mask"])
        ]
    else:
        padded_encoding["attention_mask"] = pad_mask

    return padded_encoding


def _is_batched(encoding):
    if encoding.get("input_ids") is not None:
        return isinstance(encoding["input_ids"][0], list) and isinstance(encoding["input_ids"][0][0], int)
    if encoding.get("pixel_values") is not None:
        return image_utils.is_batched(encoding["pixel_values"])
    else:
        raise NotImplementedError


class Gia2Processor(ProcessorMixin):
    r"""
    Constructs a GIA2 processor which wraps a CLIP image processor and a BERT tokenizer into a single processor.

    [`Gia2Processor`] offers all the functionalities of [`CLIPImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~Gia2Processor.__call__`] and [`~Gia2Processor.decode`] for more information.

    Args:
        image_processor ([`AutoImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`]):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    DONT_TRUNCATE_OR_PAD = {"pixel_values"}  # Or, a better name for this would be

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def _truncate_and_pad(
        self, encoding: dict, padding: Union[bool, str], truncation: Union[bool, str], max_length: Optional[int] = None
    ) -> dict:
        # If max_length is not provided, use the maximum length accepted by the model.
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        # Exclude keys that we don't want to truncate or pad.
        excluded = {key: value for key, value in encoding.items() if key in self.DONT_TRUNCATE_OR_PAD}
        encoding = {key: value for key, value in encoding.items() if key not in self.DONT_TRUNCATE_OR_PAD}

        # Apply Truncation
        if truncation in [True, "lossy"]:
            encoding = truncate(encoding, max_length, preserve=False)
        elif truncation == "preserve":
            encoding = truncate(encoding, max_length, preserve=True)
        elif truncation in [False, "do_not_truncate"]:
            pass
        else:
            raise ValueError("Invalid truncation strategy:" + str(truncation))

        # Apply Padding
        if padding in [True, "longest"]:
            target_length = max(len(seq) for sequences in encoding.values() for seq in sequences)
            encoding = pad(encoding, target_length)
        elif padding == "max_length":
            encoding = pad(encoding, max_length)
        elif padding in [False, "do_not_pad"]:
            pass
        else:
            raise ValueError("Invalid padding strategy:" + str(padding))

        # Add back the excluded keys.
        encoding.update(excluded)

        return encoding

    def __call__(
        self,
        text=None,
        images=None,
        continuous_observations=None,
        discrete_observations=None,
        image_observations=None,
        continuous_actions=None,
        discrete_actions=None,
        rewards=None,
        return_tensors=None,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to BertTokenizerFast's [`~BertTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`,
                    `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            continuous_observations (`List[List[List[float]]]`):
                The continuous observations or batch of continuous observations to be encoded.
            discrete_observations (`List[List[List[int]]]`):
                The discrete observations or batch of discrete observations to be encoded.
            image_observations (`List[List[PIL.Image.Image]]`, `List[List[np.ndarray]]`, `List[List[torch.Tensor]]`):
                The image observations or batch of image observations to be encoded.
            continuous_actions (`List[List[List[float]]]`):
                The continuous actions or batch of continuous actions to be encoded.
            discrete_actions (``List[List[int]]`):
                The discrete actions or batch of discrete actions to be encoded.
            rewards (``List[List[float]]`):
                The rewards or batch of rewards to be encoded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        # we truncate and pad ourselves so we need to pass padding=False and truncation=False to the tokenizer
        padding = kwargs.pop("padding", False)
        truncation = kwargs.pop("truncation", False)
        max_length = kwargs.pop("max_length", None)

        # Ensure that the input is batched
        if text is not None and not isinstance(text, list):
            text = [text]

        encoding = {}
        if text is not None:
            encoding["input_ids"] = self.tokenizer(text, **kwargs)["input_ids"]
        if images is not None:
            encoding["pixel_values"] = self.image_processor(images, **kwargs).pixel_values
        if continuous_observations is not None:
            encoding["continuous_observations"] = continuous_observations
        if discrete_observations is not None:
            encoding["discrete_observations"] = discrete_observations
        if image_observations is not None:
            im_obs = [torch.stack([(to_tensor(im) - 0.5) / 0.5 for im in ep]) for ep in image_observations]
            encoding["image_observations"] = im_obs
        if continuous_actions is not None:
            encoding["continuous_actions"] = continuous_actions
        if discrete_actions is not None:
            encoding["discrete_actions"] = discrete_actions
        if rewards is not None:
            encoding["rewards"] = rewards

        # Handle image+text case, need to reduce the max_len as the image and text will be concatenated
        if text is not None and images is not None:
            if max_length is None:
                max_length = self.tokenizer.model_max_length
            max_length -= (224 // 16) ** 2  # substract the number of image tokens

        encoding = self._truncate_and_pad(encoding, padding, truncation, max_length)

        # Particular case, we handle the conversion to tensor of image_observations, as the format used
        # (list of tensors) is not properly handled by the BatchEncoding class:
        if "image_observations" in encoding:
            encoding["image_observations"] = torch.stack(encoding["image_observations"])

        return BatchEncoding(encoding, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def pad(self, *args, **kwargs):
        inputs = {key: [arg[key] for arg in args[0]] for key in args[0][0].keys()}
        return self(**inputs, **kwargs)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "continuous_observations",
            "discrete_observations",
            "image_observations",
            "continuous_actions",
            "discrete_actions",
            "rewards",
        ]


Gia2Processor.register_for_auto_class("AutoProcessor")
