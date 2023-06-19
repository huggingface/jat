#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


from datasets import concatenate_datasets
from transformers import Trainer

from gia import GiaConfig, GiaModel
from gia.config import Arguments
from gia.datasets import GiaDataCollator, load_and_process_dataset


def main():
    args = Arguments.parse_args()

    config = GiaConfig.from_args(args)
    model = GiaModel(config)

    # Load, prompt and process the datasets
    train_datasets = load_and_process_dataset(args, "train", config)
    train_dataset = concatenate_datasets(list(train_datasets.values()))

    trainer = Trainer(
        model,
        args,
        data_collator=GiaDataCollator(),
        train_dataset=train_dataset,
        # eval_dataset=test_datasets,  # TODO: See https://github.com/huggingface/gia/issues/65
    )
    trainer.train()


if __name__ == "__main__":
    main()
