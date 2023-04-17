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
>>> from gia.datasets import get_dataloader
>>> from gia.config import DatasetArguments
>>> args = DatasetArguments(task_names=["babyai-go-to", "mujoco-ant"])
>>> dataloader = get_dataloader(args)
>>> iterator = iter(dataloader)
>>> batch = next(iterator)
>>> batch.keys()
dict_keys(['rewards', 'dones', 'continuous_observations', 'continuous_actions', 'continuous_observations_loss_mask',
   'continuous_actions_loss_mask', 'rewards_attention_mask', 'dones_attention_mask',
   'continuous_observations_attention_mask', 'continuous_actions_attention_mask'])
>>> batch["continuous_observations"].shape
torch.Size([8, 28, 27])
>>> batch = next(iterator)
>>> batch.keys()
dict_keys(['rewards', 'dones', 'text_observations', 'discrete_observations', 'image_observations',
    'discrete_actions', 'patches_positions', 'text_observations_loss_mask', 'discrete_observations_loss_mask',
    'image_observations_loss_mask', 'discrete_actions_loss_mask', 'rewards_attention_mask',
    'dones_attention_mask', 'text_observations_attention_mask', 'discrete_observations_attention_mask',
    'discrete_actions_attention_mask', 'image_observations_attention_mask'])
>>> batch["image_observations"].shape
torch.Size([8, 39, 16, 3, 16, 16])
```

### Embed

```python
>>> from gia.config import Arguments
>>> from gia.datasets import get_dataloader
>>> from gia.model.embedding import Embeddings
>>> args = Arguments(task_names=["babyai-go-to"], embed_dim=128)
>>> dataloader = get_dataloader(args)
>>> embeddings = Embeddings(args)
>>> batch = next(iter(dataloader))
>>> for key, value in embeddings(batch).items():
...     print(f"{key}: {value.shape} {value.dtype}")
... 
embeddings: torch.Size([8, 1014, 128]) torch.float32
loss_mask: torch.Size([8, 1014]) torch.bool
attention_mask: torch.Size([8, 1014]) torch.bool
tokens: torch.Size([8, 1014]) torch.int64
```
