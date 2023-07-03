import random

import numpy as np
import pytest

from gia.processing.processing import GiaContinuousTokenizer, GiaDiscreteTokenizer, GiaImageProcessor, GiaProcessor


def test_continuous_tokenizer():
    tokenizer = GiaContinuousTokenizer()
    input_data = [[-0.8, 0.5, 0.2], [0.1, -0.3, 0.7]]
    output = tokenizer(input_data)
    assert isinstance(output, dict)
    assert "input_ids" in output
    assert isinstance(output["input_ids"], list)
    assert len(output["input_ids"]) == len(input_data)
    for i in range(len(input_data)):
        assert len(output["input_ids"][i]) == len(input_data[i])


def test_decode_continuous():
    tokenizer = GiaContinuousTokenizer()
    tokens = [[256, 512, 768]]
    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, list)
    assert len(decoded) == 1
    assert isinstance(decoded[0], list)
    assert all(isinstance(d, float) for d in decoded[0])
    assert decoded[0][0] < 1.0
    assert abs(decoded[0][1]) < 1e-4
    assert decoded[0][2] > 1.0


def test_discrete_tokenizer():
    tokenizer = GiaDiscreteTokenizer(token_shift=10)
    input_data = [[0, 1, 2], [3, 4, 5]]
    output = tokenizer(input_data)
    assert isinstance(output, dict)
    assert "input_ids" in output
    assert isinstance(output["input_ids"], list)
    assert len(output["input_ids"]) == len(input_data)
    for i in range(len(input_data)):
        assert len(output["input_ids"][i]) == len(input_data[i])
        assert output["input_ids"][i][0] == 10 + input_data[i][0]


def test_decode_discrete():
    tokenizer = GiaDiscreteTokenizer(token_shift=10)
    tokens = [[256, 512, 768]]
    decoded = tokenizer.decode(tokens)
    assert decoded == [[246, 502, 758]]  # tokens - tokens_shift


def test_image_processor():
    processor = GiaImageProcessor()
    data = np.random.randint(0, 255, (10, 32, 32, 2), dtype=np.uint8).tolist()
    features = processor(data)
    assert isinstance(features, dict)
    assert "patches" in features
    assert "patch_positions" in features
    assert isinstance(features["patches"], list)
    assert isinstance(features["patch_positions"], list)
    for batch_idx in range(len(data)):
        assert len(features["patches"][batch_idx]) == len(features["patch_positions"][batch_idx])
        for patch_idx in range(len(features["patches"][batch_idx])):
            assert features["patches"][batch_idx][patch_idx].shape == (4, 16, 16)
            assert len(features["patch_positions"][batch_idx][patch_idx]) == 2
            assert len(features["patch_positions"][batch_idx][patch_idx][0]) == 2
            assert len(features["patch_positions"][batch_idx][patch_idx][1]) == 2


def test_extract_patches():
    processor = GiaImageProcessor(patch_size=3)
    arr = np.arange(2 * 9 * 6 * 3, dtype=np.uint8).reshape(2, 9, 6, 3)  # B, H, W, C
    batch_patches = processor(arr.tolist())["patches"]
    arr = arr.transpose(0, 3, 1, 2)  # C, H, W
    for batch_idx in range(len(batch_patches)):
        patches = batch_patches[batch_idx]
        np.testing.assert_equal(patches[0][:3], arr[batch_idx, :, 0:3, 0:3])
        np.testing.assert_equal(patches[0][:3], arr[batch_idx, :, 0:3, 0:3])
        np.testing.assert_equal(patches[1][:3], arr[batch_idx, :, 0:3, 3:6])
        np.testing.assert_equal(patches[2][:3], arr[batch_idx, :, 3:6, 0:3])
        np.testing.assert_equal(patches[3][:3], arr[batch_idx, :, 3:6, 3:6])
        np.testing.assert_equal(patches[4][:3], arr[batch_idx, :, 6:9, 0:3])
        np.testing.assert_equal(patches[5][:3], arr[batch_idx, :, 6:9, 3:6])
        for patch in patches:  # pad to have 4 channels
            np.testing.assert_equal(patch[3], np.zeros((3, 3), dtype=np.uint8))


def generate_data(batch_size, features):
    output = {}
    seq_lens = [np.random.randint(5, 10) for _ in range(batch_size)]  # only used for episode data
    if "text" in features:
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        vocab = [" ".join(random.choices(words, k=np.random.randint(10, 30))) for _ in range(100)]
        output["text"] = random.choices(vocab, k=batch_size)
    if "images" in features:
        output["images"] = [np.random.randint(0, 255, size=(84, 84, 3)).tolist() for _ in range(batch_size)]
    if "text_observations" in features:
        vocab = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        output["text_observations"] = [random.choices(vocab, k=seq_len) for seq_len in seq_lens]
    if "image_observations" in features:
        output["image_observations"] = [
            np.random.randint(0, 255, size=(seq_len, 84, 84, 3)).tolist() for seq_len in seq_lens
        ]
    if "discrete_observations" in features:
        output["discrete_observations"] = [
            np.random.randint(0, 16, size=(seq_len, 2)).tolist() for seq_len in seq_lens
        ]
    if "continuous_observations" in features:
        output["continuous_observations"] = [np.random.rand(seq_len, 2).tolist() for seq_len in seq_lens]
    if "discrete_actions" in features:
        output["discrete_actions"] = [np.random.randint(0, 16, size=(seq_len,)).tolist() for seq_len in seq_lens]
    if "continuous_actions" in features:
        output["continuous_actions"] = [np.random.rand(seq_len, 2).tolist() for seq_len in seq_lens]
    if "rewards" in features:
        output["rewards"] = [np.random.rand(seq_len).tolist() for seq_len in seq_lens]

    return output


EP_DATA_CONFIGS = [
    ["continuous_observations", "continuous_actions", "rewards"],  # mujoco and metaworld
    ["image_observations", "discrete_actions", "rewards"],  # atari
    ["text_observations", "discrete_observation", "discrete_actions", "rewards"],  # babyai
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
    data = generate_data(5, features)
    out = processor(**data, padding=False)  # equivalent to padding="do_not_pad"

    # Check that length is consistent across keys but not across batch_idx
    seq_len = []
    for sequences in out.values():
        for batch_idx in range(len(sequences)):
            if len(seq_len) <= batch_idx:
                seq_len.append(len(sequences[batch_idx]))
            assert len(sequences[batch_idx]) == seq_len[batch_idx]
    # very low chance that all the 5 sequences have the same length
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
    data = generate_data(2, ["continuous_observations", "discrete_actions"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the local positions are [0, 1, None,  None, 0, 1, None, ...]
    for sequences in out["local_positions"]:
        assert all(val == 0 for val in sequences[0::4])
        assert all(val == 1 for val in sequences[1::4])
        assert all(val is None for val in sequences[2::4])  # action
        assert all(val is None for val in sequences[3::4])  # separator


def test_gia_processor_local_positions_non_singleton_multi_modality():
    processor = GiaProcessor(local_positions_groups=[["discrete_observations", "continuous_observations"]])
    data = generate_data(2, ["discrete_observations", "continuous_observations", "discrete_actions"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the local positions are [0, 1, 2, None, 0, 1, 2, None, ...]
    for sequences in out["local_positions"]:
        assert all(val == 0 for val in sequences[0::6])  # discrete observations have 2 components
        assert all(val == 1 for val in sequences[1::6])
        assert all(val == 2 for val in sequences[2::6])  # continuous observations have 2 components
        assert all(val == 3 for val in sequences[3::6])
        assert all(val is None for val in sequences[4::6])  # action
        assert all(val is None for val in sequences[5::6])  # separator


def test_gia_processor_mask_loss_modalities_episode():
    processor = GiaProcessor(mask_loss_modalities=["continuous_observations"])
    data = generate_data(2, ["discrete_observations", "continuous_observations", "discrete_actions"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the mask is [None, False, False, True, None, ...]
    for sequences in out["loss_mask"]:
        assert all(val is None for val in sequences[0::6])  # discrete observations have 2 components
        assert all(val is None for val in sequences[1::6])
        assert all(not val for val in sequences[2::6])
        assert all(not val for val in sequences[3::6])  # continuous observations have 2 components
        assert all(val for val in sequences[4::6])  # action (not masked by default)
        assert all(val is None for val in sequences[5::6])  # separator


def test_gia_processor_mask_loss_modalities_standalone():
    processor = GiaProcessor(mask_loss_modalities=["text"])
    data = generate_data(2, ["text", "images"])
    out = processor(**data, truncation=False, padding=False)

    # Check that the mask is [None, False, None, ...]
    for sequences in out["loss_mask"]:
        assert all(val is None for val in sequences[:36])  # image (84*84 corresponds to 36 patches)
        assert all(not val for val in sequences[36:])  # text (maske)
