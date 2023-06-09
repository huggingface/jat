from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from gia.datasets import GiaDataCollator, maybe_prompt_dataset
from gia.processing import GiaProcessor


task_names = ["babyai-go-to", "mujoco-ant"]
split = "train[:3]"  # take the first 3 episodes

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

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=3, collate_fn=GiaDataCollator())

# Print the first batch
batch = next(iter(dataloader))

print(
    f"""
First sample [:15]:
    Tokens:                      {batch['input_ids'][0, :15].tolist()}
    Input type:                  {batch['input_types'][0, :15].tolist()}
    Local positions:             {batch['local_positions'][0, :15].tolist()}
    Attention mask:              {batch['attention_mask'][0, :15].tolist()}
    Loss mask:                   {batch['loss_mask'][0, :15].tolist()}

Second sample [:15]:
    Tokens:                      {batch['input_ids'][1, :15].tolist()}
    Input type:                  {batch['input_types'][1, :15].tolist()}
    Local positions:             {batch['local_positions'][1, :15].tolist()}
    Attention mask:              {batch['attention_mask'][1, :15].tolist()}
    Loss mask:                   {batch['loss_mask'][1, :15].tolist()}

Third sample [:15]:
    Tokens:                      {batch['input_ids'][2, :15].tolist()}
    Input type:                  {batch['input_types'][2, :15].tolist()}
    Local positions:             {batch['local_positions'][2, :15].tolist()}
    Attention mask:              {batch['attention_mask'][2, :15].tolist()}
    Loss mask:                   {batch['loss_mask'][2, :15].tolist()}
"""
)
