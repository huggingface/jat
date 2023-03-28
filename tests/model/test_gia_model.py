from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import load_batched_dataset
from gia.model import GiaModel


def test_gia_model():
    args = Arguments()
    model = GiaModel(args)

    dataset = load_batched_dataset("mujoco-ant")
    dataloader = DataLoader(dataset)

    batch = next(iter(dataloader))
    out = model(batch)
    assert out.loss.item() > 0.0
