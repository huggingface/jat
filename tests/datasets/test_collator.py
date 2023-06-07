import pytest
import torch
import numpy as np
from gia.datasets.collator import GIADataCollator


def test_collate_input_ids():
    collator = GIADataCollator()
    features = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_ids"}
    assert output["input_ids"].dtype == torch.int64
    assert output["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_collate_patches():
    collator = GIADataCollator()
    patches = [
        [np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8) for _ in range(3)],
        [np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8) for _ in range(3)],
    ]
    features = [
        {"patches": patches[0]},
        {"patches": patches[1]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"patches"}
    assert output["patches"].dtype == torch.uint8
    excepted = torch.from_numpy(np.array(patches))  # same as torch.tensor(...), but faster
    assert torch.all(output["patches"] == excepted)


def test_collate_patch_positions():
    collator = GIADataCollator()
    patch_positions = [
        [np.random.uniform(0, 1, (2, 2)).astype(np.float32) for _ in range(3)],
        [np.random.uniform(0, 1, (2, 2)).astype(np.float32) for _ in range(3)],
    ]
    features = [
        {"patch_positions": patch_positions[0]},
        {"patch_positions": patch_positions[1]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"patch_positions"}
    assert output["patch_positions"].dtype == torch.float32
    excepted = torch.from_numpy(np.array(patch_positions))  # same as torch.tensor(...), but faster
    assert torch.all(output["patch_positions"] == excepted)


def test_collate_input_types():
    collator = GIADataCollator()
    features = [
        {"input_types": [1, 0, 1]},
        {"input_types": [0, 1, 0]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_types"}
    assert output["input_types"].dtype == torch.int64
    assert output["input_types"].tolist() == [[1, 0, 1], [0, 1, 0]]


def test_collate_local_positions():
    collator = GIADataCollator()
    features = [
        {"local_positions": [1, 2, 3]},
        {"local_positions": [4, 5, 6]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"local_positions"}
    assert output["local_positions"].dtype == torch.int64
    assert output["local_positions"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_collate_loss_mask():
    collator = GIADataCollator()
    features = [
        {"loss_mask": [True, False, True]},
        {"loss_mask": [False, True, False]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"loss_mask"}
    assert output["loss_mask"].dtype == torch.bool
    assert output["loss_mask"].tolist() == [[True, False, True], [False, True, False]]


def test_collate_attention_mask():
    collator = GIADataCollator()
    features = [
        {"attention_mask": [True, False, True]},
        {"attention_mask": [False, True, False]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"attention_mask"}
    assert output["attention_mask"].dtype == torch.bool
    assert output["attention_mask"].tolist() == [[True, False, True], [False, True, False]]


def test_collate_multiple_features():
    collator = GIADataCollator()
    features = [
        {
            "input_ids": [1, 2, 3],
            "local_positions": [7, 8, 9],
        },
        {
            "input_ids": [4, 5, 6],
            "local_positions": [10, 11, 12],
        },
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_ids", "local_positions"}
    assert output["input_ids"].dtype == torch.int64
    assert output["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert output["local_positions"].dtype == torch.int64
    assert output["local_positions"].tolist() == [[7, 8, 9], [10, 11, 12]]


def test_exclude_empty_key():
    collator = GIADataCollator()
    features = [
        {
            "input_ids": [1, 2, 3],
            "local_positions": None,
        },
        {
            "input_ids": [4, 5, 6],
            "local_positions": None,
        },
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_ids"}
    assert output["input_ids"].dtype == torch.int64
    assert output["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]


def test_pad_none_features_sequence_input_ids():
    collator = GIADataCollator()
    features = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_ids"}
    assert output["input_ids"].dtype == torch.int64
    assert output["input_ids"].tolist() == [[1, 2, 3], [0, 0, 0]]


def test_pad_none_features_sequence_patches():
    collator = GIADataCollator()
    patches = [np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8) for _ in range(3)]
    features = [
        {"patches": patches},
        {"patches": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"patches"}
    assert output["patches"].dtype == torch.uint8
    excepted = torch.zeros((2, 3, 4, 16, 16), dtype=torch.uint8)
    excepted[0, :] = torch.from_numpy(np.array(patches))
    assert torch.all(output["patches"] == excepted)


def test_pad_none_features_sequence_patch_positions():
    collator = GIADataCollator()
    patch_positions = [np.random.uniform(0, 1, (2, 2)).astype(np.float32) for _ in range(3)]
    features = [
        {"patch_positions": patch_positions},
        {"patch_positions": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"patch_positions"}
    assert output["patch_positions"].dtype == torch.float32
    excepted = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    excepted[0, :] = torch.from_numpy(np.array(patch_positions))
    assert torch.all(output["patch_positions"] == excepted)


def test_pad_none_features_sequence_input_types():
    collator = GIADataCollator()
    features = [
        {"input_types": [1, 0, 1]},
        {"input_types": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_types"}
    assert output["input_types"].dtype == torch.int64
    assert output["input_types"].tolist() == [[1, 0, 1], [-1, -1, -1]]


def test_pad_none_features_sequence_local_positions():
    collator = GIADataCollator()
    features = [
        {"local_positions": [1, 2, 3]},
        {"local_positions": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"local_positions"}
    assert output["local_positions"].dtype == torch.int64
    assert output["local_positions"].tolist() == [[1, 2, 3], [-1, -1, -1]]


def test_pad_none_features_sequence_loss_mask():
    collator = GIADataCollator()
    features = [
        {"loss_mask": [True, False, True]},
        {"loss_mask": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"loss_mask"}
    assert output["loss_mask"].dtype == torch.bool
    assert output["loss_mask"].tolist() == [[True, False, True], [True, True, True]]


def test_pad_none_features_sequence_attention_mask():
    collator = GIADataCollator()
    features = [
        {"attention_mask": [True, False, True]},
        {"attention_mask": None},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"attention_mask"}
    assert output["attention_mask"].dtype == torch.bool
    assert output["attention_mask"].tolist() == [[True, False, True], [True, True, True]]


def test_pad_none_feature_input_ids():
    collator = GIADataCollator()
    features = [
        {"input_ids": [1, None, 3]},
        {"input_ids": [4, 5, None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_ids"}
    assert output["input_ids"].dtype == torch.int64
    assert output["input_ids"].tolist() == [[1, 0, 3], [4, 5, 0]]


def test_pad_none_feature_patches():
    collator = GIADataCollator()
    patches = [
        [np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8) for _ in range(2)],
        [np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8) for _ in range(2)],
    ]
    features = [
        {"patches": [patches[0][0], None, patches[0][1]]},
        {"patches": [patches[1][0], patches[1][1], None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"patches"}
    assert output["patches"].dtype == torch.uint8
    excepted = torch.zeros((2, 3, 4, 16, 16), dtype=torch.uint8)
    excepted[0, 0, :] = torch.from_numpy(np.array(patches[0][0]))
    excepted[0, 2, :] = torch.from_numpy(np.array(patches[0][1]))
    excepted[1, 0, :] = torch.from_numpy(np.array(patches[1][0]))
    excepted[1, 1, :] = torch.from_numpy(np.array(patches[1][1]))
    assert torch.all(output["patches"] == excepted)


def test_pad_none_feature_patch_positions():
    collator = GIADataCollator()
    patch_positions = [
        [np.random.uniform(0, 1, (2, 2)).astype(np.float32) for _ in range(2)],
        [np.random.uniform(0, 1, (2, 2)).astype(np.float32) for _ in range(2)],
    ]
    features = [
        {"patch_positions": [patch_positions[0][0], None, patch_positions[0][1]]},
        {"patch_positions": [patch_positions[1][0], patch_positions[1][1], None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"patch_positions"}
    assert output["patch_positions"].dtype == torch.float32
    excepted = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    excepted[0, 0, :] = torch.from_numpy(np.array(patch_positions[0][0]))
    excepted[0, 2, :] = torch.from_numpy(np.array(patch_positions[0][1]))
    excepted[1, 0, :] = torch.from_numpy(np.array(patch_positions[1][0]))
    excepted[1, 1, :] = torch.from_numpy(np.array(patch_positions[1][1]))
    assert torch.all(output["patch_positions"] == excepted)


def test_pad_none_feature_input_types():
    # CAUTION: Shouldn't happen in practice, as input_types should never be None
    collator = GIADataCollator()
    features = [
        {"input_types": [1, None, 0]},
        {"input_types": [0, 1, None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"input_types"}
    assert output["input_types"].dtype == torch.int64
    assert output["input_types"].tolist() == [[1, -1, 0], [0, 1, -1]]


def test_pad_none_feature_local_positions():
    collator = GIADataCollator()
    features = [
        {"local_positions": [1, None, 3]},
        {"local_positions": [4, 5, None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"local_positions"}
    assert output["local_positions"].dtype == torch.int64
    assert output["local_positions"].tolist() == [[1, -1, 3], [4, 5, -1]]


def test_pad_none_feature_loss_mask():
    collator = GIADataCollator()
    features = [
        {"loss_mask": [True, None, False]},
        {"loss_mask": [False, True, None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"loss_mask"}
    assert output["loss_mask"].dtype == torch.bool
    assert output["loss_mask"].tolist() == [[True, True, False], [False, True, True]]


def test_pad_none_feature_attention_mask():
    collator = GIADataCollator()
    features = [
        {"attention_mask": [True, None, False]},
        {"attention_mask": [False, True, None]},
    ]
    output = collator(features)
    assert isinstance(output, dict)
    assert output.keys() == {"attention_mask"}
    assert output["attention_mask"].dtype == torch.bool
    assert output["attention_mask"].tolist() == [[True, True, False], [False, True, True]]


def test_catch_invalid_key():
    collator = GIADataCollator()
    features = [
        {
            "input_ids": [1, 2, 3],
            "invalid_key": [0, 1],
        },
    ]
    with pytest.raises(KeyError):
        collator(features)


def test_catch_different_lengths_for_same_key():
    collator = GIADataCollator()
    features = [
        {
            "input_ids": [1, 2, 3],
        },
        {
            "input_ids": [4, 5],
        },
    ]
    with pytest.raises(ValueError):
        collator(features)


def test_catch_different_lengths_for_different_keys():
    collator = GIADataCollator()
    features = [
        {
            "input_ids": [1, 2, 3],
            "local_positions": [7, 8],
        },
        {
            "input_ids": [4, 5, 6],
            "local_positions": [9, 10],
        },
    ]
    with pytest.raises(ValueError):
        collator(features)
