from functools import partial

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from gato.datasets import GatoDataCollator, Prompter, needs_prompt
from gato.processing import GatoProcessor


task_names = ["babyai-go-to", "mujoco-ant"]
split = "train[:3]"  # take the first 3 episodes

processor = GatoProcessor()


# Load, prompt and process the datasets
datasets = {
    task_name: load_dataset("gia-project/gia-dataset", task_name, split=split) for task_name in task_names
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

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=3, collate_fn=GatoDataCollator())

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
