#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


from transformers import Trainer

from gia import GiaModel, GiaConfig
from gia.config import Arguments
from gia.datasets import GiaDataCollator, load_gia_dataset


def main():
    args = Arguments.parse_args()

    model_config = GiaConfig.from_args(args)
    model = GiaModel(model_config)

    # Load, prompt and process the datasets
    train_dataset = load_gia_dataset(args, model_config)  # I don't like that we have to pass two configs/args here

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
