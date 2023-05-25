from datasets import Dataset
from transformers import Trainer

from gia.config import parse_args
from gia.datasets import collate_fn, load_gia_dataset
from gia.model import GiaModel
from gia.processing import GiaProcessor

# Initialize the processor and model
args = parse_args()
processor = GiaProcessor(args)
model = GiaModel(args)

# Load the dataset
train_dataset = load_gia_dataset(task_names=args.task_names, split="train")
train_dataset = processor(**train_dataset)
train_dataset = Dataset.from_dict(train_dataset)  # <- This line
trainer = Trainer(model, args, data_collator=collate_fn, train_dataset=train_dataset)
trainer.train()
