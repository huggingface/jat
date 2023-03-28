from gia.datasets import load_batched_dataset

dataset = load_batched_dataset("babyai-go-to")

for key in dataset.keys():
    print(key, dataset[0][key].shape)
