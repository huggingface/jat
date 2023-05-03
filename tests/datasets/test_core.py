from typing import Dict

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from gia.config import DatasetArguments
from gia.datasets import load_gia_dataset

BATCH_SIZE = 128
C, H, W = 3, 16, 16
PATCH_SIZE = 8
OBS_SIZE = 4
SEQ_LEN = 32


@pytest.mark.parametrize("split", ["all", "train", "test"])
def test_load_gia_dataset(split):
    dataset = load_gia_dataset("mujoco-ant", split=split)
    assert set(dataset.keys()) == {"rewards", "continuous_observations", "continuous_actions"}

    dataloader = DataLoader(dataset)
    for idx, episode in enumerate(dataloader):
        for t in range(10):
            assert len(episode["continuous_observations"][t]) == 27
            assert len(episode["continuous_actions"][t]) == 8
            assert len(episode["rewards"][t]) == 1
        if idx == 10:
            break
