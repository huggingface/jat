from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import Trainer

from gia.config import Arguments
from gia.datasets import GIADataCollator, maybe_prompt_dataset
from gia.model import GiaModel
from gia.processing import GiaProcessor

task_names = ["atari-alien", "mujoco-ant"]
split = "train[:10]"

args = Arguments(task_names=task_names, embed_dim=12, output_dir="./")
args = Arguments(task_names=["mujoco-ant"], output_dir="./")

processor = GiaProcessor(args)

load_from_cache = True  # should be set to false the first time
if not load_from_cache:
    # Load, prompt and process the dataset
    datasets = {task_name: load_dataset("gia-project/gia-dataset", task_name, split=split) for task_name in task_names}
    datasets = {task_name: maybe_prompt_dataset(dataset) for task_name, dataset in datasets.items()}
    datasets = {
        task_name: dataset.map(lambda batch: processor(**batch), remove_columns=dataset.column_names, batched=True)
        for task_name, dataset in datasets.items()
    }
    dataset = concatenate_datasets(list(datasets.values()))
    dataset.save_to_disk("./dataset")

dataset = Dataset.load_from_disk("./dataset")

# Initialize the processor and model
model = GiaModel(args)

# Load the dataset
trainer = Trainer(model, args, data_collator=GIADataCollator(), train_dataset=dataset)
trainer.train()
