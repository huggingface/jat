from gia.config import DatasetArguments
from gia.datasets import load_batched_dataset

args = DatasetArguments()
dataset = load_batched_dataset("babyai-go-to", args)

for key in dataset.keys():
    print(key, dataset[0][key].shape)
