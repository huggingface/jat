#!/usr/bin/env python3
"""Load and generate batch for all datasets from the GIA dataset"""

import argparse
import os

from datasets import get_dataset_config_names, load_dataset
from datasets.config import HF_DATASETS_CACHE


parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", default=[])

tasks = parser.parse_args().tasks
if tasks == ["all"]:
    tasks = get_dataset_config_names("gia-project/gia-dataset")  # get all task names from gia dataset

for task in tasks:
    print(f"Loading {task}...")
    cache_path = f"{HF_DATASETS_CACHE}/gia-project/gia-dataset/{task}"
    if not os.path.exists(cache_path):
        dataset = load_dataset("gia-project/gia-dataset", task)
        dataset.save_to_disk(cache_path)
