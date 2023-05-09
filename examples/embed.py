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

print(
    f"""
First sample:
    Tokens:                {batch['input_ids'][0]}
    Patches (1st elemt):   {batch['patches'][0, :, 0, 0, 0]}
    Positions (1st elemt): {batch["positions"][0, :, 0, 0]}
    Input type:            {batch['input_type'][0]}
    Attention mask:        {batch['attention_mask'][0]}

Second sample:
    Tokens:                {batch['input_ids'][1]}
    Patches (1st elemt):   {batch['patches'][1, :, 0, 0, 0]}
    Positions (1st elemt): {batch["positions"][1, :, 0, 0]}
    Input type:            {batch['input_type'][1]}
    Attention mask:        {batch['attention_mask'][1]}

Third sample:
    Tokens:                {batch['input_ids'][2]}
    Patches (1st elemt):   {batch['patches'][2, :, 0, 0, 0]}
    Positions (1st elemt): {batch["positions"][2, :, 0, 0]}
    Input type:            {batch['input_type'][2]}
    Attention mask:        {batch['attention_mask'][2]}
"""
)
