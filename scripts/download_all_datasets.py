#!/usr/bin/env python3
"""Load and generate batch for all datasets from the GIA dataset"""

from datasets import get_dataset_config_names

from gia.config import DatasetArguments
from gia.datasets import load_batched_dataset


args = DatasetArguments()

task_names = get_dataset_config_names("gia-project/gia-dataset")  # get all task names from gia dataset
for task_name in task_names:
    print(f"Loading {task_name}...")
    load_batched_dataset(task_name, args)
