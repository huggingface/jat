import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets import load_mixed_dataset
from gia.model import GiaModel


def collate(batch):
    for sample in batch:
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])
    return batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Arguments(task_names=["babyai-go-to", "mujoco-ant"], batch_size=1)
dataset = load_mixed_dataset(args)
dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate)
model = GiaModel(args).to(device)
for batch in tqdm(dataloader):
    for sample in batch:
        for key in sample.keys():
            sample[key] = sample[key].to(device)
    out = model(batch)
    tqdm.write(f"out.loss: {str(out.loss.item())}")
