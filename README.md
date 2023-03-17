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

## Usage

### Dataset loading

Load the dataset using:

```python
>>> from datasets import load_dataset
>>> from torch.utils.data import DataLoader
>>> dataset = load_dataset("qgallouedec/gia_dataset", "metaworld-assembly-v2", split="train")
>>> dataset.set_format(type="torch")
>>> dataloader = DataLoader(dataset, batch_size=4)
>>> sample = next(iter(dataloader))
>>> print(sample.keys()) 
dict_keys(['observations', 'actions', 'rewards', 'dones'])
>>> print(sample["observations"].shape)
torch.Size([4, 39])
```

For details, see https://huggingface.co/datasets/qgallouedec/gia_dataset.

### Tokenize and embed

Example script to use the tokenizer and embedding layer.

```python
import numpy as np
import torch

from gia.model.embedding import Embeddings
from gia.tokenizers.multimodal_tokenizer import MultiModalTokenizer

# Define tokenizer and embedding layer
tokenizer = MultiModalTokenizer()
embedding_layer = Embeddings(embedding_dim=32)

# Load dataset (100k samples) TODO: update this
# First, clone it with `git clone https://huggingface.co/datasets/edbeeching/prj_gia_dataset_mujoco_ant_1111/`
dataset = np.load("prj_gia_dataset_mujoco_ant_1111/dataset.npy", allow_pickle=True)

# Convert numpy object to dict. Keys are ['observations', 'actions', 'dones', 'rewards']
dataset = dataset.item()
observations = torch.from_numpy(dataset["observations"])
actions = torch.from_numpy(dataset["actions"]).clamp(-1, 1)  # TODO: remove clamp when clamping is done in the dataset

# Tokenize and embed
tokens = tokenizer(tensors=observations, actions=actions)
print(tokens.shape)  # torch.Size([100000, 36])
embeddings = embedding_layer(tokens)
print(embeddings.shape)  # torch.Size([100000, 36, 32])
```