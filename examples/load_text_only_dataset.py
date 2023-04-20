import numpy as np
from datasets import load_dataset

dataset = load_dataset("ClementRomac/cleaned_deduplicated_oscar")
observations = np.array([dataset["train"][idx]["text"] for idx in range(len(dataset["train"]))])
dones = np.array([True for _ in range(30)])
dataset = {"text_observations": observations, "dones":dones}
