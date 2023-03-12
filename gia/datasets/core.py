import torch
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(self):
        self.task = None

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index_acc = 0
        for d in self.task_datasets:  # relatively quick for a small number of datasets
            if index_acc <= idx < len(d) + index_acc:  # can be rewriten just idx - index_acc < len(d)
                sample = d[idx - index_acc]
                if self.task is not None:
                    sample["task"] = self.task

                return sample
            index_acc += len(d)


class TaskDataset:
    pass
