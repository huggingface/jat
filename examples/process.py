import numpy as np

from gia.config import DatasetArguments
from gia.datasets import collate_fn
from torch.utils.data import DataLoader
from gia.datasets.utils import DatasetDict
from gia.processing import GiaProcessor

image_1 = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
image_2 = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
image_3 = np.random.randint(0, 255, (3, 40, 32), dtype=np.uint8)

# The dataset contains 3 samples.
#    1. An episode with 2 steps. The observation is a single integer, and the action is a single integer.
#    2. An episode with 2 steps. The observation is a image and a text, and the action is a tuple of two integers.
#    3. A standalone image with a text.

dataset = {
    "text_observations": [None, ["good", "not good"], None],
    "image_observations": [None, [image_1, image_2], None],
    "discrete_observations": [[2, 3], None, None],
    "discrete_actions": [[1, 2], [[3, 4], [5, 6]], None],
    "text": [None, None, "Hello world!"],
    "images": [None, None, image_3],
}

args = DatasetArguments()
processor = GiaProcessor(args)
dataset = DatasetDict(processor(**dataset))

dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
batch = next(iter(dataloader))

print(
    f"""
First sample [:20]:
    Tokens:                      {batch['input_ids'][0, :20].tolist()}
    Local positions:             {batch['local_positions'][0, :20].tolist()}
    Patches (1st elemt):         {batch['patches'][0, :20, 0, 0, 0].tolist()}
    Patch positions (1st elemt): {batch["patch_positions"][0, :20, 0, 0].tolist()}
    Input type:                  {batch['input_types'][0, :20].tolist()}
    Attention mask:              {batch['attention_mask'][0, :20].tolist()}
    Loss mask:                   {batch['loss_mask'][0, :20].tolist()}

Second sample [:20]:
    Tokens:                      {batch['input_ids'][1, :20].tolist()}
    Local positions:             {batch['local_positions'][1, :20].tolist()}
    Patches (1st elemt):         {batch['patches'][1, :20, 0, 0, 0].tolist()}
    Patch positions (1st elemt): {batch["patch_positions"][1, :20, 0, 0].tolist()}
    Input type:                  {batch['input_types'][1, :20].tolist()}
    Attention mask:              {batch['attention_mask'][1, :20].tolist()}
    Loss mask:                   {batch['loss_mask'][1, :20].tolist()}

Third sample [:20]:
    Tokens:                      {batch['input_ids'][2, :20].tolist()}
    Local positions:             {batch['local_positions'][2, :20].tolist()}
    Patches (1st elemt):         {batch['patches'][2, :20, 0, 0, 0].tolist()}
    Patch positions (1st elemt): {batch["patch_positions"][2, :20, 0, 0].tolist()}
    Input type:                  {batch['input_types'][2, :20].tolist()}
    Attention mask:              {batch['attention_mask'][2, :20].tolist()}
    Loss mask:                   {batch['loss_mask'][2, :20].tolist()}
"""
)