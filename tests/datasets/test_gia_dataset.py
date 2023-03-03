import pytest

from gia.datasets import GiaDataset


@pytest.mark.parametrize("seq_len", [128])
def test_gia_dataset(seq_len):
    class Args:
        pass

    args = Args()
    args.tasks = ["mujoco"]
    args.seq_len = seq_len

    dataset = GiaDataset(args)

    for sample in dataset:
        assert sample["tokens"].shape == (seq_len,)
        assert sample["attn_ids"].shape == (seq_len,)
        assert sample["local_position_ids"].shape == (seq_len,)
        assert type(sample["dataset_name"]) == str
