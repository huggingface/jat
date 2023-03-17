from gia.config import Arguments
from gia.datasets.core import MultiTaskDataset
from gia.datasets.mujoco_dataset import MujocoDataset

DATASET_CLASS_MAPPING = {
    "mujoco": MujocoDataset,
    "atari": None,
    # ...
}


class GiaDataset(MultiTaskDataset):
    def __init__(self, args: Arguments) -> None:
        super().__init__()
        self.task_datasets = []

        for task in args.tasks:
            tasks_dataset_class = DATASET_CLASS_MAPPING[task]
            self.task_datasets.append(tasks_dataset_class(task, args))

        self.dataset_len = sum(len(d) for d in self.task_datasets)


if __name__ == "__main__":
    args = Arguments()
    dataset = GiaDataset(args)

    print(dataset)
