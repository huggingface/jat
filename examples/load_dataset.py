from datasets import load_dataset
import numpy as np

# Load the dataset
dataset = load_dataset("gia-project/gia-dataset", "mujoco-ant", split="train")

print(
    f"""
Column names:                {dataset.column_names}
Number of episodes:          {len(dataset)}
Length of the first episode: {len(dataset[0]['continuous_observations'])}
First observation:           {np.round(dataset[0]['continuous_observations'][0], 1).tolist()}
First action:                {np.round(dataset[0]['continuous_actions'][0], 1).tolist()}
First reward:                {np.round(dataset[0]['rewards'][0], 1)}
"""
)
