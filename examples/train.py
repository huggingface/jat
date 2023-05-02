
from gia.datasets import collate_fn, load_mixed_dataset
from gia.config import parse_args

from gia.model import GiaModel
from transformers import TrainingArguments, Trainer

args = parse_args()
args.task_names = "mujoco-walker"
dataset = load_mixed_dataset(args)

model = GiaModel(args)
train_args = TrainingArguments(output_dir=args.save_dir, remove_unused_columns=False)

trainer = Trainer(
    model, 
    train_args,
    data_collator=collate_fn,
    train_dataset=dataset,
)

trainer.train()