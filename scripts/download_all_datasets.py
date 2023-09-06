#!/usr/bin/env python3
"""Load and generate batch for all datasets from the GIA dataset"""

from datasets import get_dataset_config_names, load_dataset


task_names = get_dataset_config_names("gia-project/gia-dataset-parquet")  # get all task names from gia dataset
for task_name in task_names:
    print(f"Loading {task_name}...")
    load_dataset("gia-project/gia-dataset-parquet", task_name)
