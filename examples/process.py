import numpy as np

from gia.config import DatasetArguments
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
    "image": [None, None, image_3],
}

args = DatasetArguments()
processor = GiaProcessor(args)
processed = processor(**dataset)

patches_first_elmt = [[patch[0][0][0] for patch in patches] for patches in processed["patches"]]
positions_first_elmt = [[round(position[0][0], 2) for position in positions] for positions in processed["positions"]]

print(
    f"""
First sample:
    Tokens:                {processed['input_ids'][0]}
    Patches (1st elemt):   {patches_first_elmt[0]}
    Positions (1st elemt): {positions_first_elmt[0]}
    Input type:            {processed['input_type'][0]}

Second sample:
    Tokens:                {processed['input_ids'][1]}
    Patches (1st elemt):   {patches_first_elmt[1]}
    Positions (1st elemt): {positions_first_elmt[1]}
    Input type:            {processed['input_type'][1]}

Third sample:
    Tokens:                {processed['input_ids'][2]}
    Patches (1st elemt):   {patches_first_elmt[2]}
    Positions (1st elemt): {positions_first_elmt[2]}
    Input type:            {processed['input_type'][2]}
"""
)
