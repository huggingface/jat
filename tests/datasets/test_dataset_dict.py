import pytest

from gia.datasets.dataset_dict import DatasetDict


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
