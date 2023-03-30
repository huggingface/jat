from typing import Dict

import numpy as np
import pytest

from gia.datasets.batch_generator import (
    generate_batch,
    get_dataloader,
    stack_with_padding,
)

BATCH_SIZE = 128
C, H, W = 3, 16, 16
PATCH_SIZE = 8
OBS_SIZE = 4
SEQ_LEN = 32


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
    batches = generate_batch(dataset, seq_len=SEQ_LEN, patch_size=PATCH_SIZE, use_separator=use_separator)

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
        "dones_attention_mask",
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
    assert batches["dones_attention_mask"].shape == (num_seq, num_int_per_seq)


def test_batch_generator_no_prompt(dataset: Dict[str, np.ndarray]):
    exp_nb_patches = (W // PATCH_SIZE) * (H // PATCH_SIZE)  # Expected number of patches per image
    num_token_per_int = OBS_SIZE + exp_nb_patches + 1 + 1
    num_int_per_seq = SEQ_LEN // num_token_per_int
    num_seq = BATCH_SIZE // num_int_per_seq

    # Generate batches
    batches = generate_batch(dataset, seq_len=SEQ_LEN, patch_size=PATCH_SIZE, p_prompt=0.0)

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
    assert batches["dones_attention_mask"].shape == (num_seq, num_int_per_seq)

    # Check values
    for i in range(num_seq):
        for key in dataset.keys():
            expected_seq = dataset[key][i * num_int_per_seq : (i + 1) * num_int_per_seq]
            assert np.all(batches[key][i] == expected_seq)


def test_empty_list():
    # Test with an empty list
    with pytest.raises(ValueError):
        stack_with_padding([])


def test_padding_value():
    # Test with a non-zero padding value
    x = [np.ones((2, 2)), np.zeros((3, 2))]
    stacked, mask = stack_with_padding(x, padding_value=-1)
    assert stacked.shape == (2, 3, 2)
    assert mask.shape == (2, 3, 2)
    target_stacked = np.array(
        [
            [[1, 1], [1, 1], [-1, -1]],
            [[0, 0], [0, 0], [0, 0]],
        ]
    )
    target_mask = np.array(
        [
            [[True, True], [True, True], [False, False]],
            [[True, True], [True, True], [True, True]],
        ]
    )
    assert np.array_equal(stacked, target_stacked)
    assert np.array_equal(mask, target_mask)


def test_same_shapes():
    # Test with arrays of same shapes
    x = [np.ones((2, 2)), np.zeros((2, 2))]
    stacked, mask = stack_with_padding(x)
    assert stacked.shape == (2, 2, 2)
    assert mask.shape == (2, 2, 2)
    target_stacked = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    target_mask = np.array([[[True, True], [True, True]], [[True, True], [True, True]]])
    assert np.array_equal(stacked, target_stacked)
    assert np.array_equal(mask, target_mask)


def test_get_dataloader():
    # dataloader = get_dataloader(["babyai-go-to", "mujoco-ant"], shuffle=True, batch_size=2)
    dataloader = get_dataloader(["mujoco-ant"], shuffle=True, batch_size=2)
    ant_keys = {
        "rewards",
        "dones",
        "continuous_observations",
        "continuous_actions",
        "continuous_observations_loss_mask",
        "continuous_actions_loss_mask",
        "rewards_attention_mask",
        "dones_attention_mask",
        "continuous_observations_attention_mask",
        "continuous_actions_attention_mask",
    }
    go_to_keys = {
        "rewards",
        "dones",
        "text_observations",
        "discrete_observations",
        "image_observations",
        "discrete_actions",
        "patches_positions",
        "text_observations_loss_mask",
        "discrete_observations_loss_mask",
        "image_observations_loss_mask",
        "discrete_actions_loss_mask",
        "rewards_attention_mask",
        "dones_attention_mask",
        "text_observations_attention_mask",
        "discrete_observations_attention_mask",
        "discrete_actions_attention_mask",
        "image_observations_attention_mask",
    }
    # ant_sampled, go_to_sampled = False, False
    ant_sampled, go_to_sampled = False, True
    for batch in dataloader:
        if set(batch.keys()) == ant_keys:
            assert batch["continuous_observations"].shape == (2, 28, 27)
            ant_sampled = True
        elif set(batch.keys()) == go_to_keys:
            assert batch["image_observations"].shape == (2, 39, 16, 3, 16, 16)
            go_to_sampled = True
        else:
            raise ValueError(f"Unexpected keys {set(batch.keys())}")

    assert ant_sampled and go_to_sampled
