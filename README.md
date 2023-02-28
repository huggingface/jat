# General Intelligent Agents
An open impementation of [GATO](https://www.deepmind.com/publications/a-generalist-agent)


dev install  (Linux)
`pip install -e .[dev]`

Steps:
1. Creation of imitation learning data on the hub filter datasets for `prj_gia*` <- we are here
2. Creation of tokenizer, model, training loop etc
3. Single env learning, e.g all Atari envs -> evaluation
4. Multi task learning, e.g Atari + MassiveText -> evaluation
5. Full tasks setting -> evaluation

More details to come!

## Tokenizer

Example script to use the tokenizer.

```python
import numpy as np
import torch

from gia.model.tokenization import Tokenizer

# Define tokenizer and embedding layer
tokenizer = Tokenizer()

# Load dataset (100k samples)
# First, clone it with `git clone https://huggingface.co/datasets/edbeeching/prj_gia_dataset_mujoco_ant_1111/`
dataset = np.load("prj_gia_dataset_mujoco_ant_1111/dataset.npy", allow_pickle=True)

# Convert numpy object to dict. Keys are ['observations', 'actions', 'dones', 'rewards']
dataset = dataset.item()
observations = torch.from_numpy(dataset["observations"])
actions = torch.from_numpy(dataset["actions"]).clamp(-1, 1)  # TODO: remove clamp when clamping is done in the dataset

# Tokenize
tokens = tokenizer(tensors=observations, actions=actions)
print(tokens.shape)  # torch.Size([100000, 36])
```