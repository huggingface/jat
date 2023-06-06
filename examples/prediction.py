import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets import collate_fn, load_gia_dataset
from gia.datasets.utils import DatasetDict
from gia.model import GiaModel
from gia.processing import GiaProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Arguments(task_names=["mujoco-ant"], batch_size=1, output_dir="./")
dataset = load_gia_dataset(args.task_names)
processor = GiaProcessor(args)
dataset = DatasetDict(processor(**dataset))
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
model = GiaModel(args).to(device)


for batch in tqdm(dataloader):
    for key in batch.keys():  # This loop can be removed when  you use accelerate
        batch[key] = batch[key].to(device)
    out = model(**batch)

    first_logits = [round(logit, 3) for logit in out.logits[0, 0, :5].tolist()]
    tqdm.write(f"Loss: {out.loss.item():.3f}    First 5 logits: {first_logits}")
