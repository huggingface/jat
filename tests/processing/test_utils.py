import numpy as np
import pytest

from gia.processing.utils import (
    _append,
    _interleave_episode,
    _interleave_standalone,
    _is_episode,
    interleave_batch,
)

PATCH_0 = np.zeros(shape=(3, 16, 16), dtype=np.uint8).tolist()  # PLACEHOLDER
PATCH_1 = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8).tolist()
PATCH_2 = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8).tolist()
PATCH_3 = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8).tolist()
PATCH_4 = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8).tolist()
PATCH_5 = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8).tolist()
PATCH_6 = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8).tolist()

POS0 = [[0.0, 0.0], [0.0, 0.0]]  # PLACEHOLDER
LEFT = [[0.0, 0.0], [1.0, 0.5]]
RIGHT = [[0.0, 0.5], [1.0, 1.0]]


def test_append_input_ids():
    batch_data = {"input_ids": [101, 102]}
    processed_data = {
        "input_ids": [1, 2],
        "patches": [PATCH_0, PATCH_0],
        "positions": [POS0, POS0],
        "input_type": [0, 0],
    }

    _append(batch_data, processed_data)

    expected_output = {
        "input_ids": [1, 2, 101, 102],
        "patches": [PATCH_0, PATCH_0, PATCH_0, PATCH_0],
        "positions": [POS0, POS0, POS0, POS0],
        "input_type": [0, 0, 0, 0],
    }

    assert processed_data == expected_output


def test_append_patches_and_positions():
    batch_data = {"patches": [PATCH_1, PATCH_2], "positions": [LEFT, RIGHT]}
    processed_data = {
        "input_ids": [1, 2],
        "patches": [PATCH_0, PATCH_0],
        "positions": [POS0, POS0],
        "input_type": [0, 0],
    }

    _append(batch_data, processed_data)

    expected_output = {
        "input_ids": [1, 2, 0, 0],
        "patches": [PATCH_0, PATCH_0, PATCH_1, PATCH_2],
        "positions": [POS0, POS0, LEFT, RIGHT],
        "input_type": [0, 0, 1, 1],
    }

    assert processed_data == expected_output


def test_interleave_episode():
    # 3 timesteps
    # 1 observation is composed of
    #     2 image patches (with the associated positions) and
    #     1 discrete observations composed of 3 ints
    sample_data = {
        "discrete_observations": {
            "input_ids": [[1, 2], [3, 4]],
        },
        "image_observations": {
            "patches": [[PATCH_1, PATCH_2], [PATCH_3, PATCH_4]],
            "positions": [[LEFT, RIGHT], [LEFT, RIGHT]],
        },
    }
    expected_output = {
        "input_ids": [0, 0, 1, 2, 0, 0, 3, 4],
        "patches": [PATCH_1, PATCH_2, PATCH_0, PATCH_0, PATCH_3, PATCH_4, PATCH_0, PATCH_0],
        "positions": [LEFT, RIGHT, POS0, POS0, LEFT, RIGHT, POS0, POS0],
        "input_type": [1, 1, 0, 0, 1, 1, 0, 0],
    }
    assert _interleave_episode(sample_data) == expected_output


def test_interleave_standalone():
    input_data = {
        "text": {"input_ids": [1, 2]},
        "image": {"patches": [PATCH_1, PATCH_2], "positions": [LEFT, RIGHT]},
    }
    expected_output = {
        "input_ids": [0, 0, 1, 2],
        "patches": [PATCH_1, PATCH_2, PATCH_0, PATCH_0],
        "positions": [LEFT, RIGHT, POS0, POS0],
        "input_type": [1, 1, 0, 0],
    }
    assert _interleave_standalone(input_data) == expected_output


def test_is_episode():
    # Test valid non-episode case
    sample_data_1 = {
        "image": {"patches": [], "positions": []},
        "text": {"input_ids": []},
    }
    assert not _is_episode(sample_data_1)

    # Test valid episode case
    sample_data_2 = {
        "image_observations": {"patches": [], "positions": []},
        "text_observations": {"input_ids": []},
        "discrete_observations": {"input_ids": []},
        "continuous_actions": {"input_ids": []},
    }
    assert _is_episode(sample_data_2)

    # Test invalid mixed case
    sample_data_3 = {
        "image": {"patches": [], "positions": []},
        "text_observations": {"input_ids": []},
        "continuous_actions": {"input_ids": []},
    }
    with pytest.raises(ValueError):
        _is_episode(sample_data_3)

    # Test case with extra keys
    sample_data_4 = {
        "image": {"patches": [], "positions": []},
        "text": {"input_ids": []},
        "extra_key": {"input_ids": []},
    }
    with pytest.raises(ValueError):
        _is_episode(sample_data_4)


def test_interleave_batch_episode():
    input_data = {
        "text_observations": {
            "input_ids": [
                None,
                [[1, 2], [3, 4], [5, 6]],
            ]
        },
        "image_observations": {
            "patches": [
                [[PATCH_1, PATCH_2], [PATCH_3, PATCH_4]],
                None,
            ],
            "positions": [
                [[LEFT, RIGHT], [LEFT, RIGHT]],
                None,
            ],
        },
        "discrete_actions": {
            "input_ids": [
                [[11, 12], [13, 14]],
                [[15, 16], [17, 18], [19, 20]],
            ]
        },
    }
    expected_output = {
        "input_ids": [
            [0, 0, 11, 12, 0, 0, 13, 14],
            [1, 2, 15, 16, 3, 4, 17, 18, 5, 6, 19, 20],
        ],
        "patches": [
            [PATCH_1, PATCH_2, PATCH_0, PATCH_0, PATCH_3, PATCH_4, PATCH_0, PATCH_0],
            [
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
                PATCH_0,
            ],
        ],
        "positions": [
            [LEFT, RIGHT, POS0, POS0, LEFT, RIGHT, POS0, POS0],
            [POS0, POS0, POS0, POS0, POS0, POS0, POS0, POS0, POS0, POS0, POS0, POS0],
        ],
        "input_type": [
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    }
    assert interleave_batch(input_data) == expected_output


def test_interleave_batch_standalone():
    input_data = {
        "text": {
            "input_ids": [None, [1, 2], [3, 4]],
        },
        "image": {
            "patches": [[PATCH_1, PATCH_2], None, [PATCH_3, PATCH_4]],
            "positions": [[LEFT, RIGHT], None, [LEFT, RIGHT]],
        },
    }
    expected_output = {
        "input_ids": [
            [0, 0],
            [1, 2],
            [0, 0, 3, 4],
        ],
        "patches": [
            [PATCH_1, PATCH_2],
            [PATCH_0, PATCH_0],
            [PATCH_3, PATCH_4, PATCH_0, PATCH_0],
        ],
        "positions": [
            [LEFT, RIGHT],
            [POS0, POS0],
            [LEFT, RIGHT, POS0, POS0],
        ],
        "input_type": [
            [1, 1],
            [0, 0],
            [1, 1, 0, 0],
        ],
    }
    assert interleave_batch(input_data) == expected_output


def test_interleave_batch_mixed():
    input_data = {
        "text": {
            "input_ids": [
                [1, 2],
                None,
            ],
        },
        "image_observations": {
            "patches": [
                None,
                [[PATCH_1, PATCH_2], [PATCH_3, PATCH_4]],
            ],
            "positions": [
                None,
                [[LEFT, RIGHT], [LEFT, RIGHT]],
            ],
        },
        "continuous_actions": {
            "input_ids": [
                None,
                [[11, 12], [13, 14]],
            ],
        },
    }
    expected_output = {
        "input_ids": [
            [1, 2],
            [0, 0, 11, 12, 0, 0, 13, 14],
        ],
        "patches": [
            [PATCH_0, PATCH_0],
            [PATCH_1, PATCH_2, PATCH_0, PATCH_0, PATCH_3, PATCH_4, PATCH_0, PATCH_0],
        ],
        "positions": [
            [POS0, POS0],
            [LEFT, RIGHT, POS0, POS0, LEFT, RIGHT, POS0, POS0],
        ],
        "input_type": [
            [0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
        ],
    }

    assert interleave_batch(input_data) == expected_output
