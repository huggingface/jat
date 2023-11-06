from functools import partial

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from gato.config import GatoConfig
from gato.datasets import GatoDataCollator, Prompter, needs_prompt
from gato.model import GatoModel
from gato.processing import GatoProcessor


task_names = ["babyai-go-to", "mujoco-ant"]
split = "train[:10]"

processor = GatoProcessor()

# Load, prompt and process the datasets
datasets = {
    task_name: load_dataset("gia-project/gia-dataset-parquet", task_name, split=split) for task_name in task_names
}
prompters = {task_name: Prompter(dataset) for task_name, dataset in datasets.items() if needs_prompt(task_name)}


def prompt_and_process(example, prompter):
    if prompter is not None:
        return processor(**prompter.prompt(example))
    else:
        return processor(**example)


datasets = {
    task_name: dataset.map(
        partial(prompt_and_process, prompter=prompters.get(task_name)),
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=100,  # lower batch size to avoid OOM
    )
    for task_name, dataset in datasets.items()
}

# Concatenate the datasets
dataset = concatenate_datasets(list(datasets.values()))

# To avoid doing the previous again
dataset.save_to_disk("./dataset")
dataset = Dataset.load_from_disk("./dataset")

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=GatoDataCollator())

config = GatoConfig()
model = GatoModel(config)

for batch in tqdm(dataloader):
    with torch.no_grad():
        output = model(**batch)
        tqdm.write(f"Batch keys: {list(batch.keys())}  Loss: {output.loss.item():.2f}")
