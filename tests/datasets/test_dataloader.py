from torch.utils.data.dataloader import DataLoader

from gia.config import Arguments
from gia.datasets import GiaDataset


def test_dataloader():
    args = Arguments()
    train_dataset = GiaDataset(args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    for batch in train_dataloader:
        assert len(batch["task"]) == args.train_batch_size
        # the next test will fail in a image patch setting, update it :)
        assert batch["tokens"].shape == (args.train_batch_size, args.seq_length)
        assert batch["attn_ids"].shape == (args.train_batch_size, args.seq_length, 2)
        assert batch["local_position_ids"].shape == (args.train_batch_size, args.seq_length)
