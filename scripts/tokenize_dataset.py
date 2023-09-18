#!/usr/bin/env python3
"""
Tokenize the Gia dataset
WARNING, this must be run sequentially otherwise there are upload issues.

"""


from gia.config import Arguments
from gia.datasets import load_and_process_dataset
from gia.processing import GiaProcessor


def main():
    args = Arguments.parse_args()
    processor = GiaProcessor.from_pretrained(
        args.config_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    for split in ["train", "test"]:
        datasets = load_and_process_dataset(args, split, processor)
        for task, dataset in datasets.items():
            dataset.push_to_hub(
                "gia/gia-dataset-tokenized-1024",
                config_name=task,
                split=split,
                max_shard_size="1GB",
            )


if __name__ == "__main__":
    main()
