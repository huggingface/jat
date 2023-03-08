import pytest

from gia.config import Arguments
from gia.datasets import GiaDataset


@pytest.mark.parametrize("seq_length", [128])
def test_gia_dataset(seq_length):

    args = Arguments()
    args.tasks = ["mujoco"]
    args.seq_length = seq_length

    dataset = GiaDataset(args)

    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["task"] == "mujoco"
        assert sample["tokens"].shape == (seq_length,)
        assert sample["attn_ids"].shape == (seq_length, 2)
        assert sample["local_position_ids"].shape == (seq_length,)
