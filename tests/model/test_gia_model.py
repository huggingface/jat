from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets.batch_generator import BatchGenerator
from gia.model import GiaModel


def test_gia_model():
    args = Arguments()
    model = GiaModel(args)

    dataset = BatchGenerator.load_batchified_dataset("mujoco-ant", load_from_cache_file=False)
    dataloader = DataLoader(dataset)

    batch = next(iter(dataloader))
    out = model(batch)
    assert out.loss.item() > 0.0
