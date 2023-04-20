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

Load the dataset for MuJoCo/Ant and BabyAI/GoTo. The returned batch contain tokens for actions and observations.

```python
>>> from torch.utils.data import DataLoader
>>> from gia.config import DatasetArguments
>>> from gia.datasets import load_mixed_dataset, collate_fn
>>> args = DatasetArguments(task_names=["babyai-go-to", "mujoco-ant"])
>>> dataset = load_mixed_dataset(args)
>>> dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn)
>>> batch = next(iter(dataloader))
>>> batch[0].keys()
dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions', 'continuous_observations_loss_mask',
   'continuous_actions_loss_mask', 'rewards_attention_mask', 'dones_attention_mask',
   'continuous_observations_attention_mask', 'continuous_actions_attention_mask'])
>>> batch[0]["continuous_observations"].shape
torch.Size([8, 28, 27])
>>> batch[1].keys()
dict_keys(['rewards', 'dones', 'text_observations', 'discrete_observations', 'image_observations',
    'discrete_actions', 'patches_positions', 'text_observations_loss_mask', 'discrete_observations_loss_mask',
    'image_observations_loss_mask', 'discrete_actions_loss_mask', 'rewards_attention_mask',
    'dones_attention_mask', 'text_observations_attention_mask', 'discrete_observations_attention_mask',
    'discrete_actions_attention_mask', 'image_observations_attention_mask'])
>>> batch[1]["image_observations"].shape
torch.Size([8, 39, 16, 3, 16, 16])
```

### Embed

```python
>>> from torch.utils.data import DataLoader
>>> from gia.config import Arguments
>>> from gia.datasets import collate_fn, load_mixed_dataset
>>> from gia.model.embedding import Embeddings
>>> args = Arguments(task_names=["babyai-go-to"], embed_dim=128)
>>> dataset = load_mixed_dataset(args)
>>> dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
>>> embeddings = Embeddings(args)
>>> batch = next(iter(dataloader))
>>> embeds = [embeddings(sample) for sample in batch]
>>> len(embeds)
8
>>> for key, value in embeds[0].items():
...     print(f"{key}: {value.shape} {value.dtype}")
... 
embeddings: torch.Size([1, 1014, 128]) torch.float32
loss_mask: torch.Size([1, 1014]) torch.bool
attention_mask: torch.Size([1, 1014]) torch.bool
tokens: torch.Size([1, 1014]) torch.int64
```

### Use the GIA model

```python
>>> from torch.utils.data import DataLoader
>>> from gia.config import Arguments
>>> from gia.datasets import collate_fn, load_mixed_dataset
>>> from gia.model import GiaModel
>>> args = Arguments(task_names=["babyai-go-to", "mujoco-ant"], batch_size=1)
>>> dataset = load_mixed_dataset(args)
>>> dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn)
>>> model = GiaModel(args)
>>> batch = next(iter(dataloader))
>>> output = model(batch)
>>> output.logits.shape
torch.Size([1, 1014, 33025])
>>> output.loss
tensor(10.7401, grad_fn=<DivBackward0>)
```
