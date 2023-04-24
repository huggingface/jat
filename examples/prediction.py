import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets import collate_fn, load_mixed_dataset
from gia.model import GiaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Arguments(task_names=["babyai-go-to", "mujoco-ant"], batch_size=1)
dataset = load_mixed_dataset(args)
dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn)
model = GiaModel(args).to(device)
for batch in tqdm(dataloader):
    for sample in batch:  # This loop can be removed when you use accelerate
        for key in sample.keys():
            sample[key] = sample[key].to(device)
    out = model(batch)
    tqdm.write(f"out.loss: {str(out.loss.item())}")
