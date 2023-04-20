from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import collate_fn, load_mixed_dataset
from gia.model import GiaModel

dataset = load_batched_dataset("mujoco-ant")
dataloader = DataLoader(dataset)
model = GiaModel(Arguments())
for batch in tqdm(dataloader):
    for sample in batch:  # This loop can be removed when you use accelerate
        for key in sample.keys():
            sample[key] = sample[key].to(device)
    out = model(batch)
    tqdm.write(f"out.loss: {str(out.loss.item())}")
