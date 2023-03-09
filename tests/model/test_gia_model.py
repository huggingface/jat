from torch.utils.data.dataloader import DataLoader

from gia.config import Arguments
from gia.datasets import GiaDataset
from gia.model import GiaModel


def test_gia_model():
    args = Arguments()
    args.tasks = ["mujoco"]

    dataset = GiaDataset(args)
    dataloader = DataLoader(dataset)
    model = GiaModel(args)

    batch = next(iter(dataloader))
    out = model(batch)
    assert out.loss.item() > 0.0
