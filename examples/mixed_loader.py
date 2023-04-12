from gia.datasets import get_dataloader
from gia.config import DatasetArguments

args = DatasetArguments(task_names=["babyai-go-to", "mujoco-ant"], shuffle=True, batch_size=2)
dataloader = get_dataloader(args)
iterator = iter(dataloader)

for _ in range(4):
    print(next(iterator).keys())
