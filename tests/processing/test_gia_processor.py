import random
from typing import List

import numpy as np
import pytest

from gia.processing import GiaProcessor


def generate_data(batch_size, features):
    output = {}
    num_timesteps = [np.random.randint(5, 10) for _ in range(batch_size)]  # only used for episode data
    if "text" in features:
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        vocab = [" ".join(random.choices(words, k=np.random.randint(10, 30))) for _ in range(100)]
        output["text"] = random.choices(vocab, k=batch_size)
    if "images" in features:
        output["images"] = [np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8) for _ in range(batch_size)]
    if "text_observations" in features:
        vocab = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        output["text_observations"] = [random.choices(vocab, k=seq_num_timeteps) for seq_num_timeteps in num_timesteps]
    if "image_observations" in features:
        output["image_observations"] = [
            [np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8) for _ in range(seq_num_timesteps)]
            for seq_num_timesteps in num_timesteps
        ]
    if "discrete_observations_single" in features:
        output["discrete_observations"] = [
            [np.random.randint(0, 16, dtype=np.int64) for _ in range(seq_num_timesteps)]
            for seq_num_timesteps in num_timesteps
        ]
    if "discrete_observations_multi" in features:
        output["discrete_observations"] = [
            [np.random.randint(0, 16, size=(2,), dtype=np.int64) for _ in range(seq_num_timesteps)]
            for seq_num_timesteps in num_timesteps
        ]
    if "continuous_observations" in features:
        output["continuous_observations"] = [
            [np.random.rand(2) for _ in range(seq_num_timesteps)] for seq_num_timesteps in num_timesteps
        ]
    if "discrete_actions_single" in features:
        output["discrete_actions"] = [
            [np.random.randint(0, 16, dtype=np.int64) for _ in range(seq_num_timesteps)]
            for seq_num_timesteps in num_timesteps
        ]
    if "discrete_actions_multi" in features:
        output["discrete_actions"] = [
            [np.random.randint(0, 16, size=(2,), dtype=np.int64) for _ in range(seq_num_timesteps)]
            for seq_num_timesteps in num_timesteps
        ]
    if "continuous_actions" in features:
        output["continuous_actions"] = [
            [np.random.rand(2) for _ in range(seq_num_timesteps)] for seq_num_timesteps in num_timesteps
        ]
    if "rewards" in features:
        output["rewards"] = [
            [random.random() for _ in range(seq_num_timesteps)] for seq_num_timesteps in num_timesteps
        ]

    return output


EP_DATA_CONFIGS = [
    ["continuous_observations", "continuous_actions", "rewards"],  # mujoco and metaworld
    ["image_observations", "discrete_actions_single", "rewards"],  # atari
    ["text_observations", "discrete_observation_multi", "discrete_actions_single", "rewards"],  # babyai
]
STANDALONE_DATA_CONFIGS = [
    ["text"],  # oscar
    ["text", "images"],  # vqvae and conceptual caption
]


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
@pytest.mark.parametrize("seq_len", [512, 1024])
def test_gia_processor_padding_max_length_none(features, seq_len):
    processor = GiaProcessor(seq_len=seq_len)
    data = generate_data(2, features)
    out = processor(**data, padding="max_length", max_length=None)

    for sequences in out.values():
        assert all(len(sequence) == seq_len for sequence in sequences)


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
@pytest.mark.parametrize("max_length", [512, 1024])
def test_gia_processor_padding_max_length_value(features, max_length):
    processor = GiaProcessor()
    data = generate_data(2, features)
    out = processor(**data, padding="max_length", max_length=max_length)

    for sequences in out.values():
        assert all(len(sequence) == max_length for sequence in sequences)


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
def test_gia_processor_padding_true(features):
    processor = GiaProcessor()
    data = generate_data(2, features)
    out = processor(**data, padding=True)  # equivalent to padding="longest"

    seq_len = None  # check that all sequences have the same length
    for sequences in out.values():
        if seq_len is None:
            seq_len = len(sequences[0])
        assert all(len(sequence) == seq_len for sequence in sequences)


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
def test_gia_processor_padding_false(features):
    processor = GiaProcessor()
    data = generate_data(2, features)
    out = processor(**data, padding=False)  # equivalent to padding="do_not_pad"

    # Check that length is consistent across keys but not across batch_idx
    seq_len = []
    for sequences in out.values():
        for batch_idx in range(len(sequences)):
            if len(seq_len) <= batch_idx:
                seq_len.append(len(sequences[batch_idx]))
            assert len(sequences[batch_idx]) == seq_len[batch_idx]
    assert not all(seq_len[0] == seq_len[batch_idx] for batch_idx in range(len(seq_len)))


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
def test_gia_processor_truncation_residual_no_need(features):
    processor = GiaProcessor()
    data = generate_data(2, features)
    out = processor(**data, truncation="residual", max_length=1024)
    out_no_truncation = processor(**data, truncation=False, max_length=1024)

    # We need to iterate since some values are arrays
    assert out.keys() == out_no_truncation.keys()
    for key in out.keys():
        assert len(out[key]) == len(out_no_truncation[key])
        for batch_idx in range(len(out[key])):
            assert len(out[key][batch_idx]) == len(out_no_truncation[key][batch_idx])
            for val, val_no_truncation in zip(out[key][batch_idx], out_no_truncation[key][batch_idx]):
                if isinstance(val, np.ndarray):
                    assert np.all(val == val_no_truncation)
                else:
                    assert val == val_no_truncation


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
def test_gia_processor_truncation_residual_with_need(features):
    processor = GiaProcessor()
    data = generate_data(2, features)
    max_length = 10
    out = processor(**data, truncation="residual", max_length=max_length)
    out_no_truncation = processor(**data, truncation=False)

    assert out.keys() == out_no_truncation.keys()
    for key in out.keys():
        assert len(out[key]) > len(out_no_truncation[key])  # batch size should have increased
        # Check that the sequence as been truncated to the max_length
        assert len(out[key][0]) == max_length
        # Check that the first max_length values are the same as the non-truncated sequence
        for idx in range(max_length):
            val = out[key][0][idx]
            val_no_truncation = out_no_truncation[key][0][idx]
            if isinstance(val, np.ndarray):
                assert np.all(val == val_no_truncation)
            else:
                assert val == val_no_truncation
        # Check that the next values are in the second sequence
        for idx in range(max_length):
            val = out[key][1][idx]
            val_no_truncation = out_no_truncation[key][0][idx + max_length]
            if isinstance(val, np.ndarray):
                assert np.all(val == val_no_truncation)
            else:
                assert val == val_no_truncation


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
def test_gia_processor_truncation_true(features):
    processor = GiaProcessor()
    data = generate_data(2, features)
    max_length = 10
    out = processor(**data, truncation=True, max_length=max_length)
    out_no_truncation = processor(**data, truncation=False)

    assert out.keys() == out_no_truncation.keys()
    for key in out.keys():
        assert len(out[key]) == 2  # batch size should be the same
        # Check that the sequence as been truncated to the max_length
        assert all(len(sequence) <= max_length for sequence in out[key])
        # Check that the first max_length values are the same as the non-truncated sequence
        for batch_idx in range(len(out[key])):
            for idx in range(max_length):
                val = out[key][batch_idx][idx]
                val_no_truncation = out_no_truncation[key][batch_idx][idx]
                if isinstance(val, np.ndarray):
                    assert np.all(val == val_no_truncation)
                else:
                    assert val == val_no_truncation


@pytest.mark.parametrize("features", EP_DATA_CONFIGS + STANDALONE_DATA_CONFIGS)
def test_gia_processor_truncation_false(features):
    # To test this one, we don't pad
    processor = GiaProcessor()
    data = generate_data(5, features)
    out = processor(**data, truncation=False, padding=False)

    # Seq len should vary across batch_idx (but should be consistent across keys)
    seq_len = []
    for sequences in out.values():
        for batch_idx in range(len(sequences)):
            if len(seq_len) <= batch_idx:
                seq_len.append(len(sequences[batch_idx]))
            assert len(sequences[batch_idx]) == seq_len[batch_idx]
    # very low chance that all the 5 sequences have the same length
    assert not all(seq_len[0] == seq_len[batch_idx] for batch_idx in range(len(seq_len)))


def test_gia_processor_local_positions_singleton_single_modality():
    processor = GiaProcessor(local_positions_groups=[["continuous_observations"]])  # only text
    data = generate_data(2, ["continuous_observations"])
    out = processor(**data, truncation=False, padding=False)

    for sequences in out["local_positions"]:  # Check that the local positions are [0, 1, None, 0, 1, None, ...]
        assert all(val == 0 for val in sequences[0::3])  # values have 2 components
        assert all(val == 1 for val in sequences[1::3])
        assert all(val is None for val in sequences[2::3])  # separator


def test_gia_processor_local_positions_singleton_multi_modality():
    processor = GiaProcessor(local_positions_groups=[["continuous_observations"]])  # only text
    data = generate_data(2, ["continuous_observations", "discrete_actions_single"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the local positions are [0, 1, None,  None, 0, 1, None, ...]
    for sequences in out["local_positions"]:
        assert all(val == 0 for val in sequences[0::4])
        assert all(val == 1 for val in sequences[1::4])
        assert all(val is None for val in sequences[2::4])  # action
        assert all(val is None for val in sequences[3::4])  # separator


def test_gia_processor_local_positions_non_singleton_multi_modality():
    processor = GiaProcessor(local_positions_groups=[["discrete_observations", "continuous_observations"]])
    data = generate_data(2, ["discrete_observations_single", "continuous_observations", "discrete_actions_single"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the local positions are [0, 1, 2, None, 0, 1, 2, None, ...]
    for sequences in out["local_positions"]:
        assert all(val == 0 for val in sequences[0::5])  # discrete observation
        assert all(val == 1 for val in sequences[1::5])  # continuous observations have 2 components
        assert all(val == 2 for val in sequences[2::5])
        assert all(val is None for val in sequences[3::5])  # action
        assert all(val is None for val in sequences[4::5])  # separator


def test_gia_processor_mask_loss_modalities():
    processor = GiaProcessor(mask_loss_modalities=["continuous_observations"])
    data = generate_data(2, ["discrete_observations_single", "continuous_observations", "discrete_actions_single"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the mask is [None, False, False, True, None, ...]
    for sequences in out["loss_mask"]:
        assert all(val is None for val in sequences[0::5])  # discrete observation
        assert all(not val for val in sequences[1::5])
        assert all(not val for val in sequences[2::5])  # continuous observations have 2 components
        assert all(val for val in sequences[3::5])  # action (not masked by default)
        assert all(val is None for val in sequences[4::5])  # separator
