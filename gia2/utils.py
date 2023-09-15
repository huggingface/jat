import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn, Tensor
from typing import Any, List, Optional


def compute_mse_loss(predicted: FloatTensor, true: FloatTensor, mask: BoolTensor, sizes: LongTensor) -> FloatTensor:
    """
    Compute the Mean Squared Error (MSE) loss between predicted and true observations, considering valid timesteps and sizes.

    Args:
        predicted (`torch.FloatTensor` of shape `(batch_size, max_seq_len, max_size)`):
            Predicted observations at the output of the model.
        true (`torch.FloatTensor` of shape `(batch_size, max_seq_len, max_size)`):
            Ground truth observations.
        mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.
        sizes (`torch.LongTensor` of shape `(batch_size,)`):
            Sizes for each example in the batch.

    Returns:
        loss (`torch.FloatTensor` of shape `(,)`):
            MSE loss between predicted and true observations.
    """
    # Initialize a mask for valid observation sizes
    size_mask = torch.zeros_like(predicted, dtype=torch.bool, device=predicted.device)

    for i, size in enumerate(sizes):
        size_mask[i, :, :size] = 1

    # Expand timestep mask and apply observation size mask
    expanded_mask = mask.unsqueeze(-1).expand_as(predicted) * size_mask

    # Mask the predicted and true observations
    masked_predicted = predicted * expanded_mask
    masked_true = true * expanded_mask

    # Compute MSE loss
    criterion = nn.MSELoss(reduction="sum")
    loss = criterion(masked_predicted, masked_true)

    # Normalize by the number of valid elements
    loss /= expanded_mask.sum()

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
