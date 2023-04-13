from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.config import ModelArguments
from gia.datasets import load_batched_dataset
from gia.model import GiaModel

dataset = load_batched_dataset("mujoco-ant")
dataloader = DataLoader(dataset)
args = ModelArguments()
model = GiaModel(args)
for batch in tqdm(dataloader):
    out = model(batch)
    tqdm.write(str(out))
