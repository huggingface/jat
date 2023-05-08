from gia.datasets.core import load_gia_dataset, collate_fn
import numpy as np
from torch.utils.data import DataLoader
import torch

dataset = load_gia_dataset("mujoco-ant")
print(
    f"""Keys: {list(dataset.keys())}
Number of episodes: {len(dataset)}
Length of the first episode: {len(dataset[0]['continuous_observations'])}
First observation: {np.round(dataset[0]['continuous_observations'][0], 1)}
First action: {np.round(dataset[0]['continuous_actions'][0], 1)}
First reward: {np.round(dataset[0]['rewards'][0], 1)}
"""
)

print("Sampling a batch...")
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
batch = next(iter(dataloader))
print(
    f"""Keys: {list(batch.keys())}
Batch size: {len(batch['continuous_observations'])}
Length of the first episode: {len(batch['continuous_observations'][0])}
First observation: {batch['continuous_observations'][0][0]}
First action: {batch['continuous_actions'][0][0]}
First reward: {batch['rewards'][0][0]}
"""
)
