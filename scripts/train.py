from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import Trainer

from gia.config import Arguments
from gia.datasets import GiaDataCollator, maybe_prompt_dataset
from gia.model import GiaModel
from gia.processing import GiaProcessor


task_names = ["babyai-go-to", "mujoco-ant"]
split = "train"
args = Arguments(
    task_names=task_names,
    output_dir="./",
    embed_dim=264,
    per_device_train_batch_size=4,
    logging_first_step=True,
    logging_steps=10,
    do_eval=True,
)
processor = GiaProcessor()

load_from_cache = False  # should be set to false the first time
if not load_from_cache:
    # Load, prompt and process the datasets
    train_datasets = {
        task_name: load_dataset("gia-project/gia-dataset", task_name, split="train") for task_name in task_names
    }
    train_datasets = {task_name: maybe_prompt_dataset(dataset) for task_name, dataset in train_datasets.items()}
    train_datasets = {
        task_name: dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
        for task_name, dataset in train_datasets.items()
    }
    train_dataset = concatenate_datasets(list(train_datasets.values()))
    train_dataset.save_to_disk("./train_dataset")


test_datasets = {
    task_name: load_dataset("gia-project/gia-dataset", task_name, split="test") for task_name in task_names
}
test_datasets = {task_name: maybe_prompt_dataset(dataset) for task_name, dataset in test_datasets.items()}
test_datasets = {
    task_name: dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
    for task_name, dataset in test_datasets.items()
}

train_dataset = Dataset.load_from_disk("./train_dataset")


# Initialize the processor and model
model = GiaModel(args)

# Load the dataset
trainer = Trainer(
    model, args, data_collator=GiaDataCollator(), train_dataset=train_dataset, eval_dataset=test_datasets
)
trainer.train()
