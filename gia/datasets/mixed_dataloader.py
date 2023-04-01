from typing import List

import torch
from torch.utils.data import DataLoader


class MixedDataLoader:
    """
    A mixed DataLoader class for sampling batches from multiple DataLoaders.

    Args:
        dataloaders (List[DataLoader]): A list of DataLoaders to sample from.
        shuffle (bool, optional): If set to True, the order in which DataLoaders are used is randomized.
            Defaults to False.

    The MixedDataLoader class takes a list of DataLoaders as input and allows sampling batches from each
    DataLoader in the order they appear in the list or in a shuffled order if specified. It supports
    iteration and provides the total number of batches available across all the DataLoaders.

    Example:

        >>> from torch.utils.data import DataLoader
        >>> import torch
        >>> dataset1 = torch.arange(10).reshape(5, 2)
        >>> dataset2 = torch.arange(10, 22).reshape(4, 3)
        >>> dataloader1 = DataLoader(dataset1, batch_size=2)
        >>> dataloader2 = DataLoader(dataset2, batch_size=2)
        >>> mixed_dataloader = MixedDataLoader([dataloader1, dataloader2], shuffle=True)
        >>> for batch in mixed_dataloader:
        ...     print(batch)
        tensor([[0, 1],
                [2, 3]])
        tensor([[4, 5],
                [6, 7]])
        tensor([[8, 9]])
        tensor([[10, 11, 12],
                [13, 14, 15]])
        tensor([[16, 17, 18],
                [19, 20, 21]])
    """

    def __init__(self, dataloaders: List[DataLoader], shuffle: bool = False) -> None:
        self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
        self.loader_idxs = torch.cat([torch.full((len(dataloader),), i) for i, dataloader in enumerate(dataloaders)])
        if shuffle:
            self.loader_idxs = self.loader_idxs[torch.randperm(len(self.loader_idxs))]

    def __iter__(self) -> "MixedDataLoader":
        return self

    def __len__(self) -> int:
        return sum(len(dataloader) for dataloader in self.dataloader_iters)

    def __next__(self):
        if len(self.loader_idxs) == 0:
            raise StopIteration
        loader_idx = self.loader_idxs[0]
        self.loader_idxs = self.loader_idxs[1:]
        return next(self.dataloader_iters[loader_idx])
