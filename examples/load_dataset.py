from gia.datasets.core import load_gia_dataset
import numpy as np

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
