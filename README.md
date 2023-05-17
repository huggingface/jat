# General Intelligent Agents

An open impementation of [GATO](https://www.deepmind.com/publications/a-generalist-agent)


dev install  (Linux)
`pip install -e .[dev]`

Steps:

1. Creation of imitation learning data on the hub filter datasets for `prj_gia*`
2. Creation of tokenizer, model, training loop etc
3. Single env learning, e.g all Atari envs -> evaluation
4. Multi task learning, e.g Atari + MassiveText -> evaluation
5. Full tasks setting -> evaluation  <- we are here

More details to come!

## Usage

### Dataset loading

See [GIA Dataset](https://huggingface.co/datasets/gia-project/gia-dataset)

### GIA model

```python
from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import DatasetDict, collate_fn, load_gia_dataset
from gia.model import GiaModel
from gia.processing import GiaProcessor

args = Arguments(task_names=["atari-alien", "mujoco-ant"], batch_size=1, output_dir="./")
processor = GiaProcessor(args)
model = GiaModel(args)

# Load the dataset
dataset = load_gia_dataset(args.task_names)

# Process the dataset
dataset = DatasetDict(processor(**dataset))

# Iterate
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
batch = next(iter(dataloader))

# Forward pass
output = model(**batch)
print(output.loss)
```
