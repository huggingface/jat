from typing import List, Tuple

import torch
from torch import BoolTensor, FloatTensor

from gia2.utils import compute_mse_loss, filter_tensor


class ContinuousDataCollator:
    """
    Collates a list of dictionaries containing continuous observations, actions, and rewards into a single batch.

    Parameters:
        max_size (int): Maximum size for each tensor in the batch.
        max_seq_len (int): Maximum sequence length for each tensor in the batch.

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

    def __init__(self, max_size: int, max_seq_len: int = 256):
        self.max_size = max_size
        self.max_seq_len = max_seq_len

    def _collate(self, sequences: List) -> Tuple[FloatTensor, BoolTensor]:
        """
        Collates a list of vectors into a single tensor, handling both nested and non-nested sequences.

        Args:
            sequences (`List`):
                Sequence of vectors. Each vector can either be of shape `(seq_len,)` or `(seq_len, size)`.

        Returns:
            collated (`torch.FloatTensor`):
                If sequences are nested, shape is `(batch_size, max_seq_len, max_size)`.
                Otherwise, shape is `(batch_size, max_seq_len)`.
                Collated tensor with zeros for padding.
            mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
                Boolean mask indicating valid timesteps.
        """
        batch_size = len(sequences)
        max_seq_len = min(max([len(sequence) for sequence in sequences]), self.max_seq_len)

        is_nested = isinstance(sequences[0][0], list)

        if is_nested:
            collated = torch.zeros(batch_size, max_seq_len, self.max_size, dtype=torch.float32)
        else:
            collated = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)

        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

        for i, sequence in enumerate(sequences):
            sequence = torch.tensor(sequence, dtype=torch.float32)[:max_seq_len]
            seq_len = sequence.shape[0]

            if is_nested:
                size = sequence.shape[1]
                collated[i, :seq_len, :size] = sequence
            else:
                collated[i, :seq_len] = sequence

            mask[i, :seq_len] = 1

        return collated, mask

    def __call__(self, batch):
        # Separate observations, actions, and rewards
        continuous_observations = [x["continuous_observations"] for x in batch]
        continuous_actions = [x["continuous_actions"] for x in batch]
        rewards = [x["rewards"] for x in batch]

        # Compute observation and action sizes
        observation_sizes = torch.tensor([len(x[0]) for x in continuous_observations], dtype=torch.long)
        action_sizes = torch.tensor([len(x[0]) for x in continuous_actions], dtype=torch.long)

        # Collate observations, actions and rewards
        continuous_observations, mask = self._collate(continuous_observations)
        continuous_actions, mask = self._collate(continuous_actions)
        rewards = self._collate(rewards)

        # Return collated data
        return {
            "continuous_observations": continuous_observations,
            "continuous_actions": continuous_actions,
            "rewards": rewards,
            "attention_mask": mask,
            "observation_sizes": observation_sizes,
            "action_sizes": action_sizes,
        }


if __name__ == "__main__":
    # Initialize the collator with max_size=10
    collator = ContinuousDataCollator(max_size=10)

    # Example 1: Batch with different sequence lengths and feature sizes
    batch1 = [
        {
            "continuous_observations": torch.rand(3, 4).tolist(),
            "continuous_actions": torch.rand(3, 5).tolist(),
            "rewards": torch.rand(3).tolist(),
        },
        {
            "continuous_observations": torch.rand(2, 6).tolist(),
            "continuous_actions": torch.rand(2, 7).tolist(),
            "rewards": torch.rand(2).tolist(),
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
