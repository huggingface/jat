from typing import Dict

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from gia.config import DatasetArguments
from gia.datasets import (
    collate_fn,
    load_batched_dataset,
    load_mixed_dataset,
    load_task_dataset,
)
from gia.datasets.core import generate_batch

BATCH_SIZE = 128
C, H, W = 3, 16, 16
PATCH_SIZE = 8
OBS_SIZE = 4
SEQ_LEN = 32


@pytest.mark.parametrize("split", ["all", "train", "test"])
def test_load_task_dataset(split):
    dataset = load_task_dataset("mujoco-ant", split=split, load_from_cache=False)
    assert set(dataset.keys()) == set(
        [
            "rewards",
            "dones",
            "continuous_observations",
            "continuous_actions",
        ]
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for idx, batch in enumerate(dataloader):
        assert batch["continuous_observations"].shape == (2, 27)
        assert batch["continuous_observations"].dtype == torch.float32
        assert batch["continuous_actions"].shape == (2, 8)
        assert batch["continuous_actions"].dtype == torch.float32
        assert batch["rewards"].shape == (2,)
        assert batch["rewards"].dtype == torch.float32
        assert batch["dones"].shape == (2,)
        assert batch["dones"].dtype == torch.bool
        if idx == 3:
            break


@pytest.fixture
def dataset():
    # Example dataset
    obs = np.random.rand(BATCH_SIZE, OBS_SIZE).astype(np.float32)
    img = np.random.randint(0, 255, size=(BATCH_SIZE, C, H, W), dtype=np.uint8)
    actions = np.random.randint(0, 10, size=(BATCH_SIZE,), dtype=np.int64)
    dones = np.random.rand(BATCH_SIZE) < 0.1
    return {"continuous_observations": obs, "image_observations": img, "discrete_actions": actions, "dones": dones}


@pytest.mark.parametrize("use_separator", [True, False])
def test_batch_generator(dataset: Dict[str, np.ndarray], use_separator: bool):
    # Generate batches
    args = DatasetArguments(seq_len=SEQ_LEN, patch_size=PATCH_SIZE, use_separator=use_separator)
    batches = generate_batch(dataset, args)

    # Check output keys
    assert set(batches.keys()) == {
        # Tokens, attention mask and loss mask for observations and actions
        "continuous_observations",
        "continuous_observations_attention_mask",
        "continuous_observations_loss_mask",
        "discrete_actions",
        "discrete_actions_attention_mask",
        "discrete_actions_loss_mask",
        # Special case for image observations, also returns patches positions
        "image_observations",
        "image_observations_attention_mask",
        "image_observations_loss_mask",
        "patches_positions",
        # Dones
        "dones",
    }
    exp_nb_patches = (W // PATCH_SIZE) * (H // PATCH_SIZE)  # Expected number of patches per image
    # tokens + patches + sperator + actions
    if use_separator:
        num_token_per_int = OBS_SIZE + exp_nb_patches + 1 + 1
    else:
        num_token_per_int = OBS_SIZE + exp_nb_patches + 1
    num_int_per_seq = SEQ_LEN // num_token_per_int  # Number of interactions per sequence

    # Check output shapes
    # We don't know how many batches there are, so we just check that it's the same for all outputs
    num_seq = len(batches["continuous_observations"])

    # Check output shapes
    assert batches["continuous_observations"].shape == (num_seq, num_int_per_seq, OBS_SIZE)
    assert batches["continuous_observations_attention_mask"].shape == (num_seq, num_int_per_seq, OBS_SIZE)
    assert batches["continuous_observations_loss_mask"].shape == (num_seq, num_int_per_seq, OBS_SIZE)
    assert batches["discrete_actions"].shape == (num_seq, num_int_per_seq, 1)
    assert batches["discrete_actions_attention_mask"].shape == (num_seq, num_int_per_seq, 1)
    assert batches["discrete_actions_loss_mask"].shape == (num_seq, num_int_per_seq, 1)
    assert batches["image_observations"].shape == (num_seq, num_int_per_seq, exp_nb_patches, C, PATCH_SIZE, PATCH_SIZE)
    assert batches["image_observations_attention_mask"].shape == (num_seq, num_int_per_seq, exp_nb_patches)
    assert batches["image_observations_loss_mask"].shape == (num_seq, num_int_per_seq, exp_nb_patches)
    assert batches["patches_positions"].shape == (num_seq, num_int_per_seq, exp_nb_patches, 2, 2)
    assert batches["dones"].shape == (num_seq, num_int_per_seq)


def test_batch_generator_no_prompt(dataset: Dict[str, np.ndarray]):
    exp_nb_patches = (W // PATCH_SIZE) * (H // PATCH_SIZE)  # Expected number of patches per image
    num_token_per_int = OBS_SIZE + exp_nb_patches + 1 + 1
    num_int_per_seq = SEQ_LEN // num_token_per_int
    num_seq = BATCH_SIZE // num_int_per_seq

    # Generate batches
    args = DatasetArguments(seq_len=SEQ_LEN, patch_size=PATCH_SIZE, p_prompt=0.0)
    batches = generate_batch(dataset, args)

    # Check output shapes
    assert batches["continuous_observations"].shape == (num_seq, num_int_per_seq, OBS_SIZE)
    assert batches["continuous_observations_attention_mask"].shape == (num_seq, num_int_per_seq, OBS_SIZE)
    assert batches["continuous_observations_loss_mask"].shape == (num_seq, num_int_per_seq, OBS_SIZE)
    assert batches["discrete_actions"].shape == (num_seq, num_int_per_seq, 1)
    assert batches["discrete_actions_attention_mask"].shape == (num_seq, num_int_per_seq, 1)
    assert batches["discrete_actions_loss_mask"].shape == (num_seq, num_int_per_seq, 1)
    assert batches["image_observations"].shape == (num_seq, num_int_per_seq, exp_nb_patches, C, PATCH_SIZE, PATCH_SIZE)
    assert batches["image_observations_attention_mask"].shape == (num_seq, num_int_per_seq, exp_nb_patches)
    assert batches["image_observations_loss_mask"].shape == (num_seq, num_int_per_seq, exp_nb_patches)
    assert batches["patches_positions"].shape == (num_seq, num_int_per_seq, exp_nb_patches, 2, 2)
    assert batches["dones"].shape == (num_seq, num_int_per_seq)

    # Check values
    for i in range(num_seq):
        for key in dataset.keys():
            expected_seq = dataset[key][i * num_int_per_seq : (i + 1) * num_int_per_seq]
            assert np.all(batches[key][i] == expected_seq)


def test_load_batched_dataset():
    args = DatasetArguments(shuffle=True, batch_size=2)
    dataset = load_batched_dataset("mujoco-ant", args)
    assert set(dataset.keys()) == {
        "rewards",
        "dones",
        "continuous_observations",
        "continuous_actions",
        "continuous_observations_loss_mask",
        "continuous_actions_loss_mask",
        "continuous_observations_attention_mask",
        "continuous_actions_attention_mask",
    }
    sample = dataset[0]
    assert sample["rewards"].shape == (28,)
    assert sample["dones"].shape == (28,)
    assert sample["continuous_observations"].shape == (28, 27)
    assert sample["continuous_actions"].shape == (28, 8)
    assert sample["continuous_observations_loss_mask"].shape == (28, 27)
    assert sample["continuous_actions_loss_mask"].shape == (28, 8)
    assert sample["continuous_observations_attention_mask"].shape == (28, 27)
    assert sample["continuous_actions_attention_mask"].shape == (28, 8)


def test_load_mixed_dataset():
    args = DatasetArguments(task_names=["metaworld-assembly", "mujoco-ant"])
    dataset = load_mixed_dataset(args)
    # It would be nice to test with two datasets with different keys, but currently
    # Atari and BabyAI are too big to run in the CI.
    dataloader = DataLoader(dataset, shuffle=False)
    expected_keys = {
        "rewards",
        "dones",
        "continuous_observations",
        "continuous_actions",
        "continuous_observations_loss_mask",
        "continuous_actions_loss_mask",
        "continuous_observations_attention_mask",
        "continuous_actions_attention_mask",
    }
    mujoco_sampled = False
    metaworld_sampled = False
    for batch in dataloader:
        assert set(batch.keys()) == expected_keys
        shape = batch["continuous_observations"].shape
        assert shape[0] == 1  # batch_size = 1
        if shape[1:] == (28, 27):
            mujoco_sampled = True
        elif shape[1:] == (23, 39):
            metaworld_sampled = True
        else:
            raise ValueError("Unexpected shape: {}".format(shape))
    assert mujoco_sampled and metaworld_sampled


@pytest.mark.parametrize("use_accelerate", [False, True])
def test_dataloading_with_collate(use_accelerate):
    # Just need to check that the collate function does not crash, and the output
    args = DatasetArguments(task_names=["mujoco-ant", "metaworld-assembly"])
    dataset = load_mixed_dataset(args)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    if use_accelerate:
        dataloader = Accelerator().prepare(dataloader)
    expected_keys = {
        "rewards",
        "dones",
        "continuous_observations",
        "continuous_actions",
        "continuous_observations_loss_mask",
        "continuous_actions_loss_mask",
        "continuous_observations_attention_mask",
        "continuous_actions_attention_mask",
    }
    for batch in dataloader:
        assert isinstance(batch, list)
        assert len(batch) <= 3  # usually 3, but sometimes less since drop_last=False
        for sample in batch:
            assert isinstance(sample, dict)
            assert set(sample.keys()) == expected_keys
            for value in sample.values():
                assert isinstance(value, torch.Tensor)
            shape = sample["continuous_observations"].shape
            assert shape in [(1, 28, 27), (1, 23, 39)]
