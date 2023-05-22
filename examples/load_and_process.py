from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import collate_fn, load_gia_dataset
from gia.datasets.utils import DatasetDict
from gia.processing import GiaProcessor

args = Arguments(task_names=["mujoco-ant"], output_dir="./")
dataset = load_gia_dataset(args.task_names)

processor = GiaProcessor(args)
dataset = DatasetDict(processor(**dataset))

dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
batch = next(iter(dataloader))

print(
    f"""
First sample [:15]:
    Tokens:                      {batch['input_ids'][0, :15].tolist()}
    Local positions:             {batch['local_positions'][0, :15].tolist()}
    Patches (1st elemt):         {batch['patches'][0, :15, 0, 0, 0].tolist()}
    Patch positions (1st elemt): {batch["patch_positions"][0, :15, 0, 0].tolist()}
    Input type:                  {batch['input_types'][0, :15].tolist()}
    Attention mask:              {batch['attention_mask'][0, :15].tolist()}
    Loss mask:                   {batch['loss_mask'][0, :15].tolist()}

Second sample [:15]:
    Tokens:                      {batch['input_ids'][1, :15].tolist()}
    Local positions:             {batch['local_positions'][1, :15].tolist()}
    Patches (1st elemt):         {batch['patches'][1, :15, 0, 0, 0].tolist()}
    Patch positions (1st elemt): {batch["patch_positions"][1, :15, 0, 0].tolist()}
    Input type:                  {batch['input_types'][1, :15].tolist()}
    Attention mask:              {batch['attention_mask'][1, :15].tolist()}
    Loss mask:                   {batch['loss_mask'][1, :15].tolist()}

Third sample [:15]:
    Tokens:                      {batch['input_ids'][2, :15].tolist()}
    Local positions:             {batch['local_positions'][2, :15].tolist()}
    Patches (1st elemt):         {batch['patches'][2, :15, 0, 0, 0].tolist()}
    Patch positions (1st elemt): {batch["patch_positions"][2, :15, 0, 0].tolist()}
    Input type:                  {batch['input_types'][2, :15].tolist()}
    Attention mask:              {batch['attention_mask'][2, :15].tolist()}
    Loss mask:                   {batch['loss_mask'][2, :15].tolist()}
"""
)
