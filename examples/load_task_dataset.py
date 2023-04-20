from gia.config import DatasetArguments
from gia.datasets import load_task_dataset

args = DatasetArguments()
dataset = load_task_dataset("babyai-go-to", args)
print(dataset.keys())
print(dataset[0])
