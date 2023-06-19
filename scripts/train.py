#!/usr/bin/env python3
"""Train a GIA model on the GIA dataset"""


from transformers import Trainer

from gia import GiaConfig, GiaModel
from gia.config import Arguments
from gia.datasets import GiaDataCollator, load_and_process_dataset
from gia.processing import GiaProcessor


def main():
    args = Arguments.parse_args()

    config = GiaConfig.from_args(args)
    processor = GiaProcessor(
        args.mu,
        args.M,
        config.nb_bins,
        config.patch_size,
        args.mask_loss_modalities,
        config.seq_len,
        args.local_positions_groups,
        config.use_separator,
    )
    model = GiaModel(config)

    # Load, prompt and process the datasets
    train_dataset = load_and_process_dataset(
        args.task_names, "train", processor, not args.overwrite_cache, args.preprocessing_num_workers
    )

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
