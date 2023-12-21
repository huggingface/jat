# Copied from dataset, with just a modif to allow n_contiguous

from copy import deepcopy
from typing import Iterator, List, Optional, TypeVar

import numpy as np
from datasets.arrow_dataset import Dataset, _interleave_map_style_datasets
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.features import Features
from datasets.features.features import _align_features, _check_if_features_can_be_aligned
from datasets.info import DatasetInfo
from datasets.iterable_dataset import CyclingMultiSourcesExamplesIterable, IterableDataset, _BaseExamplesIterable
from datasets.splits import NamedSplit
from datasets.utils import logging
from datasets.utils.py_utils import Literal


logger = logging.get_logger(__name__)


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


class RandomlyCyclingMultiSourcesExamplesIterable(CyclingMultiSourcesExamplesIterable):
    def __init__(
        self,
        ex_iterables: List[_BaseExamplesIterable],
        generator: np.random.Generator,
        probabilities: Optional[List[float]] = None,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
        n_contiguous: Optional[int] = None,
    ):
        super().__init__(ex_iterables, stopping_strategy)
        self.generator = deepcopy(generator)
        self.probabilities = probabilities
        # TODO(QL): implement iter_arrow
        self.n_contiguous = n_contiguous

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size=1000,
        p: Optional[List[float]] = None,
        n_contiguous: Optional[int] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        n_contiguous = n_contiguous or 1
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                for i in rng.choice(num_sources, size=random_batch_size, p=p):
                    for _ in range(n_contiguous):
                        yield int(i)

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(
            rng, len(self.ex_iterables), p=self.probabilities, n_contiguous=self.n_contiguous
        )

    def shuffle_data_sources(self, generator: np.random.Generator) -> "RandomlyCyclingMultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return RandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables,
            generator=generator,
            probabilities=self.probabilities,
            stopping_strategy=self.stopping_strategy,
            n_contiguous=self.n_contiguous,
        )

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "RandomlyCyclingMultiSourcesExamplesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return RandomlyCyclingMultiSourcesExamplesIterable(
            [iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables],
            self.generator,
            self.probabilities,
            self.stopping_strategy,
            n_contiguous=self.n_contiguous,
        )


def _interleave_iterable_datasets(
    datasets: List[IterableDataset],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    n_contiguous: Optional[int] = None,
) -> IterableDataset:
    """
    Interleave several iterable datasets (sources) into a single iterable dataset.
    The new iterable dataset alternates between the sources to yield examples.
    If `probabilities = None` (default) the iterable dataset will cycles through the sources in order for each next example in the iteration.
    If `probabilities` is not `None, the iterable dataset will sample a random source according to the provided probabilities for each next examples in the iteration.

    <Added version="2.4.0"/>

    Args:
        datasets (`List[IterableDataset]`): list of datasets to interleave
        probabilities (`List[float]`, optional, default None): If specified, the new iterable dataset samples
            examples from one source at a time according to these probabilities.
        seed (`int`, optional, default None): The random seed used to choose a source for each example.
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.

    Output:
        `datasets.IterableDataset`
    """
    datasets = [d._resolve_features() for d in datasets]

    # Perform checks
    _check_if_features_can_be_aligned([dset.features for dset in datasets])

    # TODO: improve this to account for a mix of ClassLabel and Value for example
    # right now it would keep the type of the first dataset in the list
    features = Features(
        {k: v for features in _align_features([dset.features for dset in datasets]) for k, v in features.items()}
    )

    ex_iterables = [d._ex_iterable for d in datasets]

    # Use cycling or random cycling of sources
    if probabilities is None:
        ex_iterable = CyclingMultiSourcesExamplesIterable(ex_iterables, stopping_strategy=stopping_strategy)
    else:
        generator = np.random.default_rng(seed)
        ex_iterable = RandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables,
            generator=generator,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
            n_contiguous=n_contiguous,
        )
    # Set new info - we update the features
    # setting the features also ensures to fill missing columns with None
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in datasets])
    else:
        info = info.copy()
    info.features = features
    # Get all the auth tokens per repository - in case the datasets come from different private repositories
    token_per_repo_id = {
        repo_id: token for dataset in datasets for repo_id, token in dataset._token_per_repo_id.items()
    }
    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=split, token_per_repo_id=token_per_repo_id)


def interleave_datasets(
    datasets: List[DatasetType],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    n_contiguous: Optional[int] = None,
) -> DatasetType:
    """
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.

    You can use this function on a list of [`Dataset`] objects, or on a list of [`IterableDataset`] objects.

        - If `probabilities` is `None` (default) the new dataset is constructed by cycling between each source to get the examples.
        - If `probabilities` is not `None`, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    The resulting dataset ends when one of the source datasets runs out of examples except when `oversampling` is `True`,
    in which case, the resulting dataset ends when all datasets have ran out of examples at least one time.

    Note for iterable datasets:

    In a distributed setup or in PyTorch DataLoader workers, the stopping strategy is applied per process.
    Therefore the "first_exhausted" strategy on an sharded iterable dataset can generate less samples in total (up to 1 missing sample per subdataset per worker).

    Args:
        datasets (`List[Dataset]` or `List[IterableDataset]`):
            List of datasets to interleave.
        probabilities (`List[float]`, *optional*, defaults to `None`):
            If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        seed (`int`, *optional*, defaults to `None`):
            The random seed used to choose a source for each example.
        info ([`DatasetInfo`], *optional*):
            Dataset information, like description, citation, etc.
            <Added version="2.4.0"/>
        split ([`NamedSplit`], *optional*):
            Name of the dataset split.
            <Added version="2.4.0"/>
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now, `first_exhausted` and `all_exhausted`.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have `max_length_datasets*nb_dataset` samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
    Returns:
        [`Dataset`] or [`IterableDataset`]: Return type depends on the input `datasets`
        parameter. `Dataset` if the input is a list of `Dataset`, `IterableDataset` if the input is a list of
        `IterableDataset`.

    Example:

        For regular datasets (map-style):

        ```python
        >>> from datasets import Dataset, interleave_datasets
        >>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
        >>> d2 = Dataset.from_dict({"a": [10, 11, 12]})
        >>> d3 = Dataset.from_dict({"a": [20, 21, 22]})
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [10, 0, 11, 1, 2, 20, 12, 10, 0, 1, 2, 21, 0, 11, 1, 2, 0, 1, 12, 2, 10, 0, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42)
        >>> dataset["a"]
        [10, 0, 11, 1, 2]
        >>> dataset = interleave_datasets([d1, d2, d3])
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> d1 = Dataset.from_dict({"a": [0, 1, 2]})
        >>> d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
        >>> d3 = Dataset.from_dict({"a": [20, 21, 22, 23, 24]})
        >>> dataset = interleave_datasets([d1, d2, d3])
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>> dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [0, 10, 20, 1, 11, 21, 2, 12, 22, 0, 13, 23, 1, 10, 24]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42)
        >>> dataset["a"]
        [10, 0, 11, 1, 2]
        >>> dataset = interleave_datasets([d1, d2, d3], probabilities=[0.7, 0.2, 0.1], seed=42, stopping_strategy="all_exhausted")
        >>> dataset["a"]
        [10, 0, 11, 1, 2, 20, 12, 13, ..., 0, 1, 2, 0, 24]
        For datasets in streaming mode (iterable):

        >>> from datasets import load_dataset, interleave_datasets
        >>> d1 = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)
        >>> d2 = load_dataset("oscar", "unshuffled_deduplicated_fr", split="train", streaming=True)
        >>> dataset = interleave_datasets([d1, d2])
        >>> iterator = iter(dataset)
        >>> next(iterator)
        {'text': 'Mtendere Village was inspired by the vision...}
        >>> next(iterator)
        {'text': "Média de débat d'idées, de culture...}
        ```
    """
    if not datasets:
        raise ValueError("Unable to interleave an empty list of datasets.")
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError(f"{stopping_strategy} is not supported. Please enter a valid stopping_strategy.")
    if dataset_type is Dataset:
        return _interleave_map_style_datasets(
            datasets, probabilities, seed, info=info, split=split, stopping_strategy=stopping_strategy
        )
    else:
        return _interleave_iterable_datasets(
            datasets,
            probabilities,
            seed,
            info=info,
            split=split,
            stopping_strategy=stopping_strategy,
            n_contiguous=n_contiguous,
        )
