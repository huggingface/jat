from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import collate_fn, load_gia_dataset
from gia.datasets.utils import DatasetDict
from gia.processing import GiaProcessor

args = Arguments(task_names=["mujoco-ant"], embed_dim=128)
dataset = load_gia_dataset(args.task_names)

processor = GiaProcessor(args)
dataset = DatasetDict(processor(**dataset))

dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
batch = next(iter(dataloader))

print(batch)
