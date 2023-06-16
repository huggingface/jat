#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


from transformers import Trainer

from gia import GiaModelConfig, GiaModel
from gia.datasets import GiaDataCollator, load_gia_dataset
from gia.config.arguments import parse_args


def main():
    args = parse_args()
    model_config = GiaModelConfig.from_args(args)

    model = GiaModel(model_config)

    # Load, prompt and process the datasets
    train_dataset = load_gia_dataset(args, model_config)  # I don't like that we have to pass two configs/args here
    # Initialize the model
    model_config = GiaModelConfig()
    model = GiaModel(model_config)

    # Load the dataset
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
