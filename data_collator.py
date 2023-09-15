import torch

import torch
import torch.nn as nn


class ContinuousDataCollator:
    """
    Collates a list of dictionaries containing continuous observations, actions, and rewards into a single batch.

    Parameters:
        max_size (int): Maximum size for each tensor in the batch.

    Input:
        A list of `batch_size` dictionaries, each containing tensors with the following shapes:
            - "continuous_observations": (seq_len, action_size)
            - "continuous_actions": (seq_len, observation_size)
            - "rewards": (seq_len,)

    Output:
        A dictionary containing tensors with the following shapes:
            - "continuous_observations": (batch_size, max_seq_len, max_size)
            - "continuous_actions": (batch_size, max_seq_len, max_size)
            - "rewards": (batch_size, max_seq_len)
            - "attention_mask": A tensor of shape (batch_size, max_seq_len), where 1s indicate valid timesteps and
                0s indicate padding.
            - "observation_size": A tensor of shape (batch_size,) containing the observation size for each example.
            - "action_size": A tensor of shape (batch_size,) containing the action size for each example.
        where `max_seq_len` is the maximum sequence length among all examples in the batch.
    """

    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, batch):
        batch_size = len(batch)

        # Find the max sequence length in the batch
        max_seq_len = max([x["continuous_observations"].shape[0] for x in batch])

        # Initialize tensors with zeros for padding
        continuous_observations = torch.zeros(batch_size, max_seq_len, self.max_size, dtype=torch.float32)
        continuous_actions = torch.zeros(batch_size, max_seq_len, self.max_size, dtype=torch.float32)
        rewards = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

        observation_size = torch.zeros(batch_size, dtype=torch.int64)
        action_size = torch.zeros(batch_size, dtype=torch.int64)

        # Populate tensors with data
        for i, example in enumerate(batch):
            seq_len = example["continuous_observations"].shape[0]

            continuous_observations[i, :seq_len, : example["continuous_observations"].shape[1]] = example[
                "continuous_observations"
            ]
            continuous_actions[i, :seq_len, : example["continuous_actions"].shape[1]] = example["continuous_actions"]
            rewards[i, :seq_len] = example["rewards"]

            mask[i, :seq_len] = 1

            observation_size[i] = example["continuous_observations"].shape[1]
            action_size[i] = example["continuous_actions"].shape[1]

        return {
            "continuous_observations": continuous_observations,
            "continuous_actions": continuous_actions,
            "rewards": rewards,
            "attention_mask": mask,
            "observation_sizes": observation_size,
            "action_sizes": action_size,
        }


def compute_mse_loss(predicted, true, mask, sizes):
    """
    Compute the MSE loss between predicted and true continuous observations.

    Parameters:
        predicted (Tensor): Predicted observations, shape (batch_size, max_seq_len, max_size).
        true (Tensor): True observations, shape (batch_size, max_seq_len, max_size).
        mask (Tensor): Mask indicating valid timesteps, shape (batch_size, max_seq_len).
        size (Tensor): Size for each example, shape (batch_size,).

    Returns:
        float: The MSE loss.
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


def filter_tensor(tensor, mask, sizes):
    """
    Parameters:
        tensor (Tensor): Tensor, shape (batch_size, seq_len, ...).
        mask (Tensor): Mask indicating valid timesteps, shape (batch_size, seq_len).
        sizes (Tensor): Observation size for each example, shape (batch_size,).

    Returns:
        list of list of values
    """
    batch_size, seq_len = mask.shape
    nested_list = []

    for i in range(batch_size):
        batch_list = []
        for j in range(seq_len):
            if mask[i, j].item() == 1:
                obs_size = sizes[i].item()
                values = tensor[i, j, :obs_size].tolist()
                batch_list.append(values)
        nested_list.append(batch_list)

    return nested_list


if __name__ == "__main__":
    # Initialize the collator with max_size=10
    collator = ContinuousDataCollator(max_size=10)

    # Example 1: Batch with different sequence lengths and feature sizes
    batch1 = [
        {
            "continuous_observations": torch.rand(3, 4),
            "continuous_actions": torch.rand(3, 5),
            "rewards": torch.rand(3),
        },
        {
            "continuous_observations": torch.rand(2, 6),
            "continuous_actions": torch.rand(2, 7),
            "rewards": torch.rand(2),
        },
    ]
    result1 = collator(batch1)
    print("Example 1:", result1)

    # Example 2: Batch with same sequence lengths but different feature sizes
    batch2 = [
        {
            "continuous_observations": torch.rand(4, 4),
            "continuous_actions": torch.rand(4, 5),
            "rewards": torch.rand(4),
        },
        {
            "continuous_observations": torch.rand(4, 6),
            "continuous_actions": torch.rand(4, 7),
            "rewards": torch.rand(4),
        },
    ]
    result2 = collator(batch2)
    print("Example 2:", result2)

    # Example 3: Batch with both same sequence lengths and feature sizes
    batch3 = [
        {
            "continuous_observations": torch.rand(5, 5),
            "continuous_actions": torch.rand(5, 5),
            "rewards": torch.rand(5),
        },
        {
            "continuous_observations": torch.rand(5, 5),
            "continuous_actions": torch.rand(5, 5),
            "rewards": torch.rand(5),
        },
    ]
    result3 = collator(batch3)
    print("Example 3:", result3)

    # Example usage
    predicted = torch.rand(2, 4, 10)  # Replace with your predicted observations
    true = torch.rand(2, 4, 10)  # Replace with your true observations
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)  # Replace with your mask
    size = torch.tensor([4, 6])  # Replace with your observation sizes

    loss = compute_mse_loss(predicted, true, mask, size)
    print("MSE Loss:", loss.item())

    # Example usage
    tensor = torch.rand(2, 3, 5)  # Replace with your tensor
    mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.int)  # Replace with your mask
    sizes = torch.tensor([3, 4])  # Replace with your sizes

    filtered_data = filter_tensor(tensor, mask, sizes)
    print("Filtered Data:", filtered_data)
    print(tensor)
