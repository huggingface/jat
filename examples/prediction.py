import torch
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets import get_dataloader
from gia.model import GiaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Arguments(task_names=["babyai-go-to", "mujoco-ant"], batch_size=1)
dataloader = get_dataloader(args)
model = GiaModel(args).to(device)
for batch in tqdm(dataloader):
    for key in batch.keys():
        batch[key] = batch[key].to(device)
    out = model(batch)
    tqdm.write(f"out.loss: {str(out.loss.item())}")
