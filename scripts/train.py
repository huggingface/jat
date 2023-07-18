#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


from datasets import concatenate_datasets
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, Trainer

from gia.config import Arguments
from gia.datasets import GiaDataCollator, load_and_process_dataset
from gia.eval.callback import EvaluateCheckpointCallback
from gia.eval.utils import is_slurm_available
from gia.train.scheduler import get_cosine_schedule_with_linear_warmup


def main():
    args = Arguments.parse_args()

    config = AutoConfig.from_pretrained(
        args.config_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    model = AutoModel.from_config(config=config)

    # Load, prompt and process the datasets
    train_datasets = load_and_process_dataset(args, args.train_split, config)
    train_dataset = concatenate_datasets(list(train_datasets.values()))
    test_datasets = load_and_process_dataset(args, args.test_split, config)
    if args.max_eval_samples is not None:
        test_datasets = {
            task_name: dataset.select(range(args.max_eval_samples)) for task_name, dataset in test_datasets.items()
        }

    # Load the trainer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_linear_warmup(
        optimizer, num_warmup_steps=15_000, num_decay_steps=1_000_000, final_value=1e-5
    )
    trainer = Trainer(
        model,
        args,
        data_collator=GiaDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=test_datasets,
        callbacks=[EvaluateCheckpointCallback] if args.auto_eval and is_slurm_available() else [],
        optimizers=(optimizer, scheduler),
    )
    trainer.train()


if __name__ == "__main__":
    main()
