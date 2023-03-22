from typing import Any, Dict, Sized

from torch.utils.data import Dataset


class DatasetDict(dict, Dataset):
    """
    A dataset that contains multiple fields.

    Example:
    >>> dataset = DatasetDict({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> dataset[0]
    {'a': 1, 'b': 4}
    >>> dataset[1]
    {'a': 2, 'b': 5}
    >>> dataset["b"]
    [4, 5, 6]
    >>> len(dataset)
    3
    >>> dataset.pop("a")
    [1, 2, 3]
    >>> dataset[0]
    {'b': 4}
    >>> dataset[:-1]
    {'b': [4, 5]}
    >>> len(dataset)
    3
    >>> dataset.keys()
    dict_keys(['b'])
    >>> dataset.values()
    dict_values([[4, 5, 6]])
    >>> dataset.items()
    dict_items([('b', [4, 5, 6])])
    >>> list(dataset)
    [{'b': 4}, {'b': 5}, {'b': 6}]
    """

    def __getitem__(self, key_or_index) -> Dict[str, Any]:
        if isinstance(key_or_index, (int, slice)):
            return {key: value[key_or_index] for key, value in self.items()}
        else:
            return super().__getitem__(key_or_index)

    def __len__(self) -> int:
        return len(next(iter(self.values())))

    def __iter__(self) -> Dict[str, Any]:
        for index in range(len(self)):
            yield self[index]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
