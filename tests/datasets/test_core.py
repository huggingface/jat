import random

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from gia.datasets import collate_fn, generate_prompts, load_gia_dataset

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


@pytest.fixture
def example_dataset():
    return Dataset.from_dict(
        {"text": [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]]}
    )


def test_num_prompts(example_dataset):
    num_prompts = 5
    prompts = generate_prompts(example_dataset, num_prompts)
    assert len(prompts) == num_prompts, "Number of generated prompts should match num_prompts"


def test_max_prompt_len(example_dataset):
    num_prompts = 5
    max_prompt_len = 3
    prompts = generate_prompts(example_dataset, num_prompts, max_prompt_len=max_prompt_len)
    for prompt in prompts:
        assert len(prompt["text"]) <= max_prompt_len, "Generated prompts should not exceed max_prompt_len"


def test_random_seed(example_dataset):
    random.seed(42)
    num_prompts = 5
    prompts_1 = generate_prompts(example_dataset, num_prompts, p_end=0.5)

    random.seed(42)
    prompts_2 = generate_prompts(example_dataset, num_prompts, p_end=0.5)

    assert prompts_1.to_dict() == prompts_2.to_dict(), "Using the same random seed should generate the same prompts"


def test_p_end(example_dataset):
    num_prompts = 1000
    p_end = 0.8
    prompts = generate_prompts(example_dataset, num_prompts, p_end=p_end, max_prompt_len=5)
    count_from_end = sum(prompt["text"][-1] == example_dataset["text"][0][-1] for prompt in prompts)
    expected_count_from_end = int(num_prompts * p_end)

    # We allow a small tolerance in the percentage to account for randomness
    tolerance = 0.1
    assert (
        abs(count_from_end - expected_count_from_end) / num_prompts <= tolerance
    ), "The proportion of generated prompts from the end of the dataset should be close to p_end"


def test_output_structure(example_dataset):
    prompts = generate_prompts(example_dataset, num_prompts=5)

    assert isinstance(prompts, Dataset), "Output should be a Dataset object"
    assert set(prompts.column_names) == set(
        example_dataset.column_names
    ), "Output dataset should have the same column names as the input dataset"
    assert all(isinstance(prompt["text"], list) for prompt in prompts), "Each prompt should be a list"


def test_collate_fn_same_keys():
    batch = [
        {"key1": [1, 2], "key2": [3, 4, 5]},
        {"key1": [6], "key2": [7, 8]},
    ]

    output = collate_fn(batch)
    expected_output = {
        "key1": [torch.tensor([1, 2]), torch.tensor([6])],
        "key2": [torch.tensor([3, 4, 5]), torch.tensor([7, 8])],
    }

    for key in output:
        assert key in expected_output
        for i in range(len(output[key])):
            assert torch.all(output[key][i] == expected_output[key][i])


def test_collate_fn_different_keys():
    batch = [
        {"key1": [1, 2]},
        {"key2": [3, 4, 5]},
    ]

    output = collate_fn(batch)
    expected_output = {
        "key1": [torch.tensor([1, 2]), None],
        "key2": [None, torch.tensor([3, 4, 5])],
    }

    for key in output:
        assert key in expected_output
        for i in range(len(output[key])):
            if output[key][i] is None:
                assert expected_output[key][i] is None
            else:
                assert torch.all(output[key][i] == expected_output[key][i])


def test_collate_fn_mixed_keys():
    batch = [
        {"key1": [1, 2], "key2": [3, 4, 5]},
        {"key1": [6], "key3": [7, 8]},
    ]

    output = collate_fn(batch)
    expected_output = {
        "key1": [torch.tensor([1, 2]), torch.tensor([6])],
        "key2": [torch.tensor([3, 4, 5]), None],
        "key3": [None, torch.tensor([7, 8])],
    }

    for key in output:
        assert key in expected_output
        for i in range(len(output[key])):
            if output[key][i] is None:
                assert expected_output[key][i] is None
            else:
                assert torch.all(output[key][i] == expected_output[key][i])
