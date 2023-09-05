import numpy as np
import pytest

from gia.datasets.to_hub import add_dataset_to_hub  # Replace 'your_module' with the name of the actual module.


def test_add_dataset_to_hub():
    # Testing with valid inputs
    rewards = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    continuous_observations = [np.array([0.5, 0.2, 0.8], dtype=np.float32)]
    discrete_actions = [np.array([0, 1, 1], dtype=np.int64)]

    add_dataset_to_hub(
        domain="domain1",
        task="task1",
        continuous_observations=continuous_observations,
        rewards=rewards,
        discrete_actions=discrete_actions,
        push_to_hub=False,
    )

    # Testing with no observations
    with pytest.raises(AssertionError):
        add_dataset_to_hub(
            domain="domain1", task="task1", rewards=rewards, discrete_actions=discrete_actions, push_to_hub=False
        )

    # Testing with no actions
    with pytest.raises(AssertionError):
        add_dataset_to_hub(
            domain="domain1",
            task="task1",
            continuous_observations=continuous_observations,
            rewards=rewards,
            push_to_hub=False,
        )

    # Testing with invalid rewards
    with pytest.raises(AssertionError):
        add_dataset_to_hub(
            domain="domain1",
            task="task1",
            continuous_observations=continuous_observations,
            rewards=[np.array([1, 2, 3])],  # rewards are not float32
            discrete_actions=discrete_actions,
            push_to_hub=False,
        )

    # Testing with invalid test_split
    with pytest.raises(AssertionError):
        add_dataset_to_hub(
            domain="domain1",
            task="task1",
            continuous_observations=continuous_observations,
            rewards=rewards,
            discrete_actions=discrete_actions,
            test_split=1.5,  # test_split is not in range (0, 1)
            push_to_hub=False,
        )
