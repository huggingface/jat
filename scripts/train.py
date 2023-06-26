#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


from datasets import concatenate_datasets
from transformers import AutoConfig, AutoModel, Trainer

from gia.config import Arguments
from gia.datasets import GiaDataCollator, load_and_process_dataset


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
    train_datasets = load_and_process_dataset(args, "train", config)
    train_dataset = concatenate_datasets(list(train_datasets.values()))
    test_datasets = load_and_process_dataset(args, "test", config)
    if args.max_eval_samples is not None:
        test_datasets = {
            task_name: dataset.select(range(args.max_eval_samples)) for task_name, dataset in test_datasets.items()
        }

    # Load the trainer
    trainer = Trainer(
        model,
        args,
        data_collator=GiaDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=test_datasets,
    )
    trainer.train()


if __name__ == "__main__":
    main()
