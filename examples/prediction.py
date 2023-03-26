from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets import BatchGenerator
from gia.model import GiaModel

dataset = BatchGenerator.load_batchified_dataset("mujoco-ant")
dataloader = DataLoader(dataset)
model = GiaModel(Arguments())
for batch in tqdm(dataloader):
    out = model(batch)
    tqdm.write(str(out))
