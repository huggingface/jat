from gia.datasets import collate_fn, load_mixed_dataset
from gia.config import parse_args

from gia.model import GiaModel
from transformers import Trainer

args = parse_args()
args.task_names = "mujoco"
args.remove_unused_columns = False
dataset = load_mixed_dataset(args)

model = GiaModel(args)

trainer = Trainer(
    model,
    args,
    data_collator=collate_fn,
    train_dataset=dataset,
)

trainer.train()
