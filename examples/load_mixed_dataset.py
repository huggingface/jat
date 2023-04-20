from torch.utils.data import DataLoader

from gia.config import DatasetArguments
from gia.datasets import load_mixed_dataset

args = DatasetArguments(task_names=["babyai-go-to", "mujoco-ant"])
dataset = load_mixed_dataset(args)
dataloader = DataLoader(dataset, shuffle=True, collate_fn=lambda x: x)

for idx, batch in enumerate(dataloader):
    print(f"Batch {idx}")
    for sample in batch:
        for key, value in sample.items():
            print(f"  {key}: {value.shape}, {value.dtype}")
