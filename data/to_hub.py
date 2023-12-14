import tempfile
from typing import List

import numpy as np
from huggingface_hub import HfApi, upload_folder


def add_dataset_to_hub(
    domain: str,
    task: str,
    revision: str = "main",
    test_split=0.1,
    push_to_hub=False,
    text_observations: List[np.array] = None,
    image_observations: List[np.array] = None,
    discrete_observations: List[np.array] = None,
    continuous_observations: List[np.array] = None,
    discrete_actions: List[np.array] = None,
    continuous_actions: List[np.array] = None,
    rewards: List[np.array] = None,
):
    """
    This function takes different types of observations, actions, and rewards, and prepares them for upload to the hub.
    It then optionally pushes them to a specified dataset repository on the data hub.

    Args:
        domain (str): The domain to which the dataset belongs.
        task (str): The specific task within the domain.
        revision (str, optional): The revision name. Default is "main".
        test_split (float, optional): Fraction of the dataset to be used as test data. Default is 0.1.
        push_to_hub (bool, optional): If True, the dataset will be pushed to the data hub. Default is False.
        text_observations (List[np.array], optional): List of numpy arrays with text observations.
        image_observations (List[np.array], optional): List of numpy arrays with image observations.
        discrete_observations (List[np.array], optional): List of numpy arrays with discrete observations.
        continuous_observations (List[np.array], optional): List of numpy arrays with continuous observations.
        discrete_actions (List[np.array], optional): List of numpy arrays with discrete actions.
        continuous_actions (List[np.array], optional): List of numpy arrays with continuous actions.
        rewards (List[np.array], optional): List of numpy arrays with rewards.
    """
    n_obs = 0
    n_act = 0

    assert rewards is not None  # assume we always have rewards
    assert rewards[0].dtype == np.float32, f"rewards are the of type np.float32 {rewards[0].dtype}"
    assert 0.0 < test_split < 1.0
    n_episodes = len(rewards)
    train_index_end = int(n_episodes * (1.0 - test_split))  # Assume no need to shuffle the data
    dataset_train = {}
    dataset_test = {}

    dataset_train["rewards"] = np.array(rewards[:train_index_end], dtype=object)
    if test_split > 0.0:
        dataset_test["rewards"] = np.array(rewards[train_index_end:], dtype=object)
    # check the types of the arrays
    if text_observations is not None:
        assert text_observations[0].dtype == str, f"rewards are the of type np.float32 {rewards[0].dtype}"
        n_obs += 1
        dataset_train["text_observations"] = np.array(text_observations[:train_index_end], dtype=object)
        if test_split > 0.0:
            dataset_test["text_observations"] = np.array(text_observations[train_index_end:], dtype=object)

    if image_observations is not None:
        assert (
            image_observations[0].dtype == np.uint8
        ), f"image_observations are the of type np.uint8 {image_observations[0].dtype}"
        assert len(image_observations) == n_episodes
        n_obs += 1
        dataset_train["image_observations"] = np.array(image_observations[:train_index_end], dtype=object)
        if test_split > 0.0:
            dataset_test["image_observations"] = np.array(image_observations[train_index_end:], dtype=object)

    if discrete_observations is not None:
        assert (
            discrete_observations[0].dtype == np.int64
        ), f"discrete_observations are the of type np.int64 {discrete_observations[0].dtype}"
        assert len(discrete_observations) == n_episodes
        n_obs += 1
        dataset_train["discrete_observations"] = np.array(discrete_observations[:train_index_end], dtype=object)
        if test_split > 0.0:
            dataset_test["discrete_observations"] = np.array(discrete_observations[train_index_end:], dtype=object)

    if continuous_observations is not None:
        assert (
            continuous_observations[0].dtype == np.float32
        ), f"continuous_observations are the of type np.float32 {continuous_observations[0].dtype}"
        assert len(continuous_observations) == n_episodes
        n_obs += 1
        dataset_train["continuous_observations"] = np.array(continuous_observations[:train_index_end], dtype=object)
        if test_split > 0.0:
            dataset_test["continuous_observations"] = np.array(continuous_observations[train_index_end:], dtype=object)

    if discrete_actions is not None:
        assert (
            discrete_actions[0].dtype == np.int64
        ), f"discrete_actions are the of type np.int64 {discrete_actions[0].dtype}"
        assert len(discrete_actions) == n_episodes
        n_act += 1
        dataset_train["discrete_actions"] = np.array(discrete_actions[:train_index_end], dtype=object)
        if test_split > 0.0:
            dataset_test["discrete_actions"] = np.array(discrete_actions[train_index_end:], dtype=object)

    if continuous_actions is not None:
        assert (
            continuous_actions[0].dtype == np.float32
        ), f"continuous_actions are the of type np.float32 {continuous_actions[0].dtype}"
        assert len(continuous_actions) == n_episodes
        n_act += 1
        dataset_train["continuous_actions"] = np.array(continuous_actions[:train_index_end], dtype=object)
        if test_split > 0.0:
            dataset_test["continuous_actions"] = np.array(continuous_actions[train_index_end:], dtype=object)

    assert n_obs > 0, "there must be at least one observation array"
    assert n_act > 0, "there must be at least one action array"  # TODO: loosen this contraint for text based envs?

    if not push_to_hub:  # useful for testing other logic
        return

    with tempfile.TemporaryDirectory() as tmpdirname:
        np.savez_compressed(f"{tmpdirname}/train.npz", **dataset_train)
        if test_split > 0.0:
            np.savez_compressed(f"{tmpdirname}/test.npz", **dataset_test)

        path_in_repo = f"data/{domain}/{task}/"
        commit_message = f"adds {domain} {task} {n_episodes=}"
        HfApi().create_repo(repo_id="jat-project/jat-dataset", private=False, exist_ok=True, repo_type="dataset")

        upload_folder(
            repo_id="jat-project/jat-dataset",
            commit_message=commit_message,
            folder_path=tmpdirname,
            path_in_repo=path_in_repo,
            ignore_patterns=[".git/*"],
            repo_type="dataset",
            revision=revision,
            create_pr=True,
        )


if __name__ == "__main__":
    import pytest

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
