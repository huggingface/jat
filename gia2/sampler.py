import random
from typing import Dict, Optional, Sized

from torch.utils.data import BatchSampler


class MyBatchSampler(BatchSampler):
    """
    Sampler to sample batches of data from multiple while keeping the same dataset for each batch.

    Args:
        dataset_dict (`List[int]`):
            Dictionary of datasets.
        batch_size (`int`, **optional**):
            Batch size. Just omit this argument if you want to set it later.
        weights (`Optional[List[int]]`, **optional**):
            Weights for each dataset. The keys of `weights` must be a subset of the keys of `dataset_dict`.
            If None, all datasets are considered equal.
    """

    def __init__(
        self,
        dataset_dict: Dict[str, Sized],
        batch_size: Optional[int] = None,
        weights: Optional[Dict[str, int]] = None,
    ):
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.weights = weights if weights is not None else {}

    def __iter__(self):
        # Create a list of indices for each dataset
        indices_list = []
        cum_sum = 0

        for key, dataset in self.dataset_dict.items():
            size = len(dataset)
            weight = self.weights.get(key, 1.0)
            sublist = random.choices(range(cum_sum, cum_sum + size), k=int(weight * size))
            # random.shuffle(sublist)
            indices_list.append(sublist)
            cum_sum += size

        # Create a list of batches
        batches = []
        for indices in indices_list:
            # Create batches of size self.batch_size
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i : i + self.batch_size])

        # Shuffle the batches
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        num_samples = [int(len(dataset) * self.weights.get(key, 1.0)) for key, dataset in self.dataset_dict.items()]
        num_batches = sum([s // self.batch_size + (s % self.batch_size != 0) for s in num_samples])
        return num_batches


if __name__ == "__main__":
    import torch
    from torch.utils.data import ConcatDataset, TensorDataset

    # Create a list of datasets
    datasets = {
        "a": TensorDataset(torch.arange(10)),
        "b": TensorDataset(torch.arange(10, 15)),
    }

    # Create a batch sampler
    batch_sampler = MyBatchSampler(datasets, batch_size=4)

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(ConcatDataset(datasets.values()), batch_sampler=batch_sampler)

    # Iterate over the dataloader
    for batch in dataloader:
        print(batch)
