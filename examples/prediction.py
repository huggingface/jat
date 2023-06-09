import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.config import Arguments
from gia.datasets import GIADataCollator, maybe_prompt_dataset
from gia.model import GiaModel
from gia.processing import GiaProcessor


task_names = ["babyai-go-to", "mujoco-ant"]
split = "train[:10]"

processor = GiaProcessor()

# Load, prompt and process the dataset
datasets = {task_name: load_dataset("gia-project/gia-dataset", task_name, split=split) for task_name in task_names}
datasets = {task_name: maybe_prompt_dataset(dataset) for task_name, dataset in datasets.items()}
datasets = {
    task_name: dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
    for task_name, dataset in datasets.items()
}

# Concatenate the datasets
dataset = concatenate_datasets(list(datasets.values()))

# To avoid doing the previous again
dataset.save_to_disk("./dataset")
dataset = Dataset.load_from_disk("./dataset")

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=GIADataCollator())

model = GiaModel(Arguments(embed_dim=12, output_dir="./"))

for batch in tqdm(dataloader):
    with torch.no_grad():
        output = model(**batch)
        tqdm.write(f"Batch keys: {list(batch.keys())}  Loss: {output.loss.item():.2f}")
