#!/usr/bin/env python3
"""Load and generate batch for all datasets from the JAT dataset"""

import argparse
import os

from datasets import get_dataset_config_names, load_dataset
from datasets.config import HF_DATASETS_CACHE

from jat.eval.rl.core import TASK_NAME_TO_ENV_ID


parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", default=[])

tasks = parser.parse_args().tasks
if tasks == ["all"]:
    tasks = get_dataset_config_names("jat-project/jat-dataset-tokenized")  # get all task names from jat dataset

for domain in ["atari", "babyai", "metaworld", "mujoco"]:
    if domain in tasks:
        tasks.remove(domain)
        tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

for task in tasks:
    print(f"Loading {task}...")
    cache_path = f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}"
    if not os.path.exists(cache_path):
        dataset = load_dataset("jat-project/jat-dataset-tokenized", task)
        dataset.save_to_disk(cache_path)
