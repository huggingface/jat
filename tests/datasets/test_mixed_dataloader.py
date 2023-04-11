from itertools import chain

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from gia.datasets.mixed_dataloader import MixedDataLoader


def create_dataloaders():
    dataset1 = TensorDataset(torch.randn(10, 3, 8, 8), torch.randint(0, 5, (10,)))
    dataset2 = TensorDataset(torch.randn(15, 3, 8, 8), torch.randint(0, 5, (15,)))
    dataloader1 = DataLoader(dataset1, batch_size=3)
    dataloader2 = DataLoader(dataset2, batch_size=3)
    return [dataloader1, dataloader2]


def test_mixed_data_loader_length():
    dataloaders = create_dataloaders()
    mixed_dataloader = MixedDataLoader(dataloaders)
    assert len(mixed_dataloader) == sum([len(dataloader) for dataloader in dataloaders])


def test_mixed_data_loader_iterates_over_batches():
    dataloaders = create_dataloaders()
    mixed_dataloader = MixedDataLoader(dataloaders)
    batch_count = 0
    for batch in mixed_dataloader:
        batch_count += 1
    assert batch_count == len(mixed_dataloader)


def test_mixed_data_loader_iterates_in_order():
    # When shuffle=False, the order of DataLoaders should be the same as the order in which they are
    # passed to the MixedDataLoader
    dataloaders = create_dataloaders()
    mixed_dataloader = MixedDataLoader(dataloaders, shuffle=False)

    # Check that the order of DataLoaders is the same from the order in which they are passed to the
    # MixedDataLoader
    ref_iter = chain(iter(dataloaders[0]), iter(dataloaders[1]))
    all_equal = all(
        torch.equal(a, b) for batch, ref_batch in zip(mixed_dataloader, ref_iter) for a, b in zip(batch, ref_batch)
    )
    if not all_equal:
        pytest.fail("MixedDataLoader did not iterate in the order of DataLoaders")


def test_mixed_data_loader_shuffle():
    dataloaders = create_dataloaders()
    mixed_dataloader = MixedDataLoader(dataloaders, shuffle=True)

    # Check that the order of DataLoaders is different from the order in which they are passed to the
    # MixedDataLoader
    ref_iter = chain(iter(dataloaders[0]), iter(dataloaders[1]))
    all_equal = all(
        torch.equal(a, b) for batch, ref_batch in zip(mixed_dataloader, ref_iter) for a, b in zip(batch, ref_batch)
    )
    if all_equal:
        pytest.fail("MixedDataLoader did not shuffle the order of DataLoaders")


def test_multiple_epochs():
    dataloaders = create_dataloaders()
    mixed_dataloader = MixedDataLoader(dataloaders, shuffle=True)
    count = 0
    for epoch in range(2):
        for batch in mixed_dataloader:
            count += 1

    assert count == 2 * len(mixed_dataloader)


def test_accelerate_compatibillity():
    dataloaders = create_dataloaders()
    mixed_dataloader = MixedDataLoader(dataloaders, shuffle=True)
    accelerator = Accelerator()
    dataloader = accelerator.prepare(mixed_dataloader)
    for batch in dataloader:
        pass
