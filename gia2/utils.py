from typing import Any, List, Optional

import cv2
import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn


def compute_mse_loss(predicted: FloatTensor, true: FloatTensor, mask: BoolTensor) -> FloatTensor:
    """
    Compute the Mean Squared Error (MSE) loss between predicted and true observations, considering valid timesteps.

    Args:
        predicted (`torch.FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Predicted observations at the output of the model.
        true (`torch.FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Ground truth observations.
        mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.

    Returns:
        loss (`torch.FloatTensor` of shape `(,)`):
            MSE loss between predicted and true observations.
    """
    # Expand timestep mask and apply observation size mask
    expanded_mask = mask.unsqueeze(-1).expand_as(predicted)

    # Mask the predicted and true observations
    masked_predicted = predicted * expanded_mask
    masked_true = true * expanded_mask

    # Compute MSE loss
    criterion = nn.MSELoss(reduction="sum")
    loss = criterion(masked_predicted, masked_true)

    # Normalize by the number of valid elements
    loss /= expanded_mask.sum()

    return loss


def compute_ce_loss(predicted: torch.FloatTensor, true: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
    """
    Compute the Cross Entropy (CE) loss between predicted logits and true labels, considering valid timesteps.

    Args:
        predicted (`torch.FloatTensor` of shape `(batch_size, max_seq_len, num_classes)`):
            Predicted logits at the output of the model.
        true (`torch.LongTensor` of shape `(batch_size, max_seq_len)`):
            Ground truth integer labels.
        mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.

    Returns:
        loss (`torch.FloatTensor` of shape `(,)`):
            CE loss between predicted logits and true labels.
    """
    # Flatten the tensors to fit the loss function's expected input shapes
    flat_predicted = predicted.view(-1, predicted.size(-1))
    flat_true = true.view(-1)
    flat_mask = mask.view(-1)

    # Compute CE loss
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = criterion(flat_predicted, flat_true)

    # Apply the mask to the losses
    masked_losses = losses * flat_mask.float()

    # Compute the mean loss over the masked elements
    loss = masked_losses.sum() / flat_mask.float().sum()

    return loss


def filter_tensor(
    tensor: Tensor, mask: Optional[BoolTensor] = None, sizes: Optional[LongTensor] = None
) -> List[List[Any]]:
    """
    Filters a tensor based on a mask and sizes, and returns a nested list of values.

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, seq_len, ...)`):
            Input tensor to be filtered.
        mask (`Optional[torch.BoolTensor]` of shape `(batch_size, seq_len)`, **optional**):
            Boolean mask indicating valid timesteps. If None, all timesteps are considered valid.
        sizes (`Optional[torch.LongTensor]` of shape `(batch_size,)`, **optional**):
            Observation size for each example in the batch. If None, all sizes are considered valid.

    Returns:
        `List[List[Any]]`:
            A nested list containing filtered values, considering only valid timesteps and sizes.

    Examples:
        >>> tensor = torch.arange(12).reshape(2, 3, 2)
        >>> mask = torch.tensor([[True, True, False], [True, False, False]])
        >>> filter_tensor(tensor, mask)
        [[[0, 1], [2, 3]], [[6, 7]]]
        >>> sizes = torch.tensor([2, 1])
        >>> filter_tensor(tensor, sizes=sizes)
        [[[0, 1], [2, 3], [4, 5]], [[6], [8], [10]]]
    """
    batch_size, seq_len = tensor.shape[:2]
    nested_list = []

    for i in range(batch_size):
        batch_list = []
        for j in range(seq_len):
            if mask is None or mask[i, j].item() == 1:
                obs_size = sizes[i].item() if sizes is not None else tensor.shape[-1]
                values = tensor[i, j, :obs_size].tolist()
                batch_list.append(values)
        nested_list.append(batch_list)

    return nested_list


def cyclic_expand_dim(tensor: Tensor, expanded_dim_size: int) -> Tensor:
    """
    Expands the last dimension of a tensor cyclically to a specified size.

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, seq_len, ...)`):
            Input tensor whose last dimension is to be expanded cyclically.
        expanded_dim_size (`int`):
            The desired size of the last dimension after expansion.

    Returns:
        `torch.Tensor` of shape `(batch_size, seq_len, expanded_dim_size)`:
            A tensor with its last dimension expanded cyclically to the specified size.

    Examples:
        >>> tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> cyclic_expand_dim(tensor, 5)
        tensor([[[1, 2, 1, 2, 1], [3, 4, 3, 4, 3]], [[5, 6, 5, 6, 5], [7, 8, 7, 8, 7]]])
    """
    B, L, X = tensor.shape
    indices = torch.arange(expanded_dim_size) % X
    return tensor[..., indices]


def write_video(frames: List[np.ndarray], filename: str, fps: int):
    """
    Writes a list of frames into a video file.

    Args:
        frames (`List[np.ndarray]`):
            List of frames in RGB format.
        filename (`str`):
            Output video filename including the extension.
        fps (`int`):
            Frames per second for the output video.
    """
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    shape = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(filename, fourcc, fps, shape)

    # Write frames to video
    for frame in frames:
        out.write(frame[..., [2, 1, 0]])  # convert RGB to BGR and write

    # Release resources
    out.release()
