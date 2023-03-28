# General Intelligent Agents

An open impementation of [GATO](https://www.deepmind.com/publications/a-generalist-agent)


dev install  (Linux)
`pip install -e .[dev]`

Steps:

1. Creation of imitation learning data on the hub filter datasets for `prj_gia*`
2. Creation of tokenizer, model, training loop etc
3. Single env learning, e.g all Atari envs -> evaluation  <- we are here
4. Multi task learning, e.g Atari + MassiveText -> evaluation
5. Full tasks setting -> evaluation

More details to come!

## Usage

### Dataset loading

#### Load raw values

Load the dataset using:

```python
>>> from datasets import load_dataset
>>> from torch.utils.data import DataLoader
>>> dataset = load_dataset("gia-project/gia-dataset", "metaworld-assembly-v2", split="train")
>>> dataset.set_format(type="torch")
>>> dataloader = DataLoader(dataset, batch_size=4)
>>> sample = next(iter(dataloader))
>>> print(sample.keys()) 
dict_keys(['observations', 'actions', 'rewards', 'dones'])
>>> print(sample["observations"].shape)
torch.Size([4, 39])
```

For details, see https://huggingface.co/datasets/gia-project/gia-dataset.

#### Loading batched dataset

Load the dataset for MuJoCo/Ant. The returned batch contain tokens for actions and observations.

```python
>>> from gia.datasets.batch_generator import load_batched_dataset
>>> from torch.utils.data import DataLoader
>>> dataset = load_batched_dataset("mujoco-ant", seq_len=72)
>>> dataloader = DataLoader(dataset, batch_size=1)
>>> batch = next(iter(dataloader))
>>> batch.keys()
dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions', 'continuous_observations_loss_mask', 'continuous_actions_loss_mask', 'rewards_attention_mask', 'dones_attention_mask', 'continuous_observations_attention_mask', 'continuous_actions_attention_mask'])
>>> batch["continuous_observations"].shape
torch.Size([1, 2, 27])
>>> batch["continuous_actions"]
tensor([[[32480, 32367, 32321, 32584, 32687, 32431, 32732, 32683],
         [32405, 32634, 32500, 32363, 32337, 32665, 32701, 32616]]])
```

### Load and embed

```python
from torch.utils.data import DataLoader
from gia.datasets import load_batched_dataset
from gia.model.embedding import Embeddings

dataset = load_batched_dataset("babyai-go-to")
dataloader = DataLoader(dataset)
embeddings = Embeddings()
batch = next(iter(dataloader))
emb = embeddings(batch)
for key, value in emb.items():
    print(f"{key}: {value.shape} {value.dtype}")
```
