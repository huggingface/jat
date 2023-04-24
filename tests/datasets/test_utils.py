import numpy as np
import pytest

from gia.datasets.utils import (
    DatasetDict,
    is_continuous,
    is_discrete,
    is_image,
    is_text,
    stack_with_padding,
)


@pytest.fixture
def dataset_dict() -> DatasetDict:
    return DatasetDict({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_getitem_with_int_index(dataset_dict: DatasetDict) -> None:
    assert dataset_dict[0] == {"a": 1, "b": 4}
    assert dataset_dict[1] == {"a": 2, "b": 5}


def test_getitem_with_slice_index(dataset_dict: DatasetDict) -> None:
    assert dataset_dict[:-1] == {"a": [1, 2], "b": [4, 5]}


def test_getitem_with_str_key(dataset_dict: DatasetDict) -> None:
    assert dataset_dict["a"] == [1, 2, 3]
    assert dataset_dict["b"] == [4, 5, 6]


def test_len(dataset_dict: DatasetDict) -> None:
    assert len(dataset_dict) == 3


def test_pop(dataset_dict: DatasetDict) -> None:
    assert dataset_dict.pop("a") == [1, 2, 3]
    assert dataset_dict[0] == {"b": 4}


def test_keys(dataset_dict: DatasetDict) -> None:
    assert all(key in dataset_dict.keys() for key in ["a", "b"])
    assert all(key in ["a", "b"] for key in dataset_dict.keys())


def test_values(dataset_dict: DatasetDict) -> None:
    assert all(value in dataset_dict.values() for value in [[1, 2, 3], [4, 5, 6]])
    assert all(value in [[1, 2, 3], [4, 5, 6]] for value in dataset_dict.values())


def test_items(dataset_dict: DatasetDict) -> None:
    assert list(dataset_dict.items()) == [("a", [1, 2, 3]), ("b", [4, 5, 6])]


def test_iter(dataset_dict: DatasetDict) -> None:
    assert list(dataset_dict) == [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]


def test_is_text():
    assert is_text(["hello", "world"])
    assert not is_text(["hello", 1])


def test_is_image():
    assert is_image(np.zeros((1, 1, 1, 1)))
    assert not is_image(np.zeros((1, 1)))
    assert not is_image(np.zeros((1, 1, 1)))


def test_is_continuous():
    assert is_continuous(np.zeros((1, 1), dtype=np.float32))
    assert is_continuous(np.zeros((1, 1), dtype=np.float64))
    assert not is_continuous(np.zeros((1, 1), dtype=np.int32))


def test_is_discrete():
    assert is_discrete(np.zeros((1, 1), dtype=np.int8))
    assert is_discrete(np.zeros((1, 1), dtype=np.uint16))
    assert not is_discrete(np.zeros((1, 1), dtype=np.float32))


def test_stask_empty_list():
    # Test with an empty list
    with pytest.raises(ValueError):
        stack_with_padding([])


def test_stack_padding_value():
    # Test with a non-zero padding value
    x = [np.ones((2, 2)), np.zeros((3, 2))]
    stacked = stack_with_padding(x, padding_value=-1)
    assert stacked.shape == (2, 3, 2)
    target_stacked = np.array(
        [
            [[1, 1], [1, 1], [-1, -1]],
            [[0, 0], [0, 0], [0, 0]],
        ]
    )
    assert np.array_equal(stacked, target_stacked)


def test_stack_same_shapes():
    # Test with arrays of same shapes
    x = [np.ones((2, 2)), np.zeros((2, 2))]
    stacked = stack_with_padding(x)
    assert stacked.shape == (2, 2, 2)
    target_stacked = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    assert np.array_equal(stacked, target_stacked)
