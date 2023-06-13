import random

import pytest
from datasets import Dataset

from gia.datasets.core import Prompter


@pytest.fixture
def example_dataset():
    return Dataset.from_dict(
        {"text": [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]]}
    )


def test_num_prompts(example_dataset):
    prompter = Prompter(example_dataset)
    num_prompts = 5
    prompts = prompter.generate_prompts(num_prompts)
    assert len(prompts["text"]) == num_prompts, "Number of generated prompts should match num_prompts"


def test_prompts_meet_min_length_requirement(example_dataset):
    min_prompt_len = 14
    prompter = Prompter(example_dataset, min_prompt_len=min_prompt_len, max_prompt_len=50)
    prompts = prompter.generate_prompts(num_prompts=50)
    for prompt in prompts["text"]:
        # Since the only episode is 19 steps long, the generated prompts should be at least 14 steps.
        assert len(prompt) >= min_prompt_len, "Generated prompts should not be shorter than min_prompt_len"


def test_prompts_restricted_to_episode_length(example_dataset):
    min_prompt_len = 26
    prompter = Prompter(example_dataset, min_prompt_len=min_prompt_len, max_prompt_len=50)
    prompts = prompter.generate_prompts(num_prompts=50)
    for prompt in prompts["text"]:
        # Since the only episode is 19 steps long, the generated prompts should be 19 steps long when min_prompt_len is
        # set to a higher value than 19
        assert len(prompt) == 19, "Generated prompts should be 19 steps long"


def test_max_prompt_len(example_dataset):
    max_prompt_len = 3
    prompter = Prompter(example_dataset, max_prompt_len=max_prompt_len)
    prompts = prompter.generate_prompts(num_prompts=5)
    for prompt in prompts["text"]:
        assert len(prompt) <= max_prompt_len, "Generated prompts should not exceed max_prompt_len"


def test_random_seed(example_dataset):
    prompter = Prompter(example_dataset, p_end=0.5, max_prompt_len=10)

    random.seed(42)
    num_prompts = 5
    prompts_1 = prompter.generate_prompts(num_prompts)

    random.seed(42)
    prompts_2 = prompter.generate_prompts(num_prompts)

    random.seed(43)
    prompts_3 = prompter.generate_prompts(num_prompts)

    assert prompts_1 == prompts_2, "Using the same random seed should generate the same prompts"
    assert prompts_1 != prompts_3, "Using different random seeds should generate different prompts"


def test_p_end(example_dataset):
    p_end = 0.8
    prompter = Prompter(example_dataset, p_end=p_end, max_prompt_len=5)
    num_prompts = 1000
    prompts = prompter.generate_prompts(num_prompts)
    end_element = example_dataset["text"][0][-1]
    count_from_end = sum(prompt[-1] == end_element for prompt in prompts["text"])
    expected_count_from_end = int(num_prompts * p_end)

    # We allow a small tolerance in the percentage to account for randomness
    tolerance = 0.1
    assert (
        abs(count_from_end - expected_count_from_end) / num_prompts <= tolerance
    ), "The proportion of generated prompts from the end of the dataset should be close to p_end"


def test_output_structure(example_dataset):
    prompter = Prompter(example_dataset)

    prompts = prompter.generate_prompts(num_prompts=5)

    assert isinstance(prompts, dict), "Output should be a Dataset object"
    assert "text" in prompts, "Output dataset should have the same keys as the input dataset"
    assert all(isinstance(prompt, list) for prompt in prompts["text"]), "Each prompt should be a list"
