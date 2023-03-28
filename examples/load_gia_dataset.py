from gia.datasets import load_gia_dataset

dataset = load_gia_dataset("babyai-go-to")
print(dataset.keys())
print(dataset[0])