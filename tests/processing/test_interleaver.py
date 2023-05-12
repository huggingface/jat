import numpy as np
import pytest

from gia.processing.interleaver import Interleaver

PPAD = np.zeros(shape=(4, 16, 16), dtype=np.uint8)  # PAD
P1 = np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8)
P2 = np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8)
P3 = np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8)
P4 = np.random.randint(0, 255, (4, 16, 16), dtype=np.uint8)

POS_PAD = [[0.0, 0.0], [0.0, 0.0]]  # PAD
LEFT = [[0.0, 0.0], [1.0, 0.5]]
RIGHT = [[0.0, 0.5], [1.0, 1.0]]


def test_dict_append_input_ids():
    interleaver = Interleaver()
    batch_data = {"input_ids": [101, 102]}
    processed_data = {
        "input_ids": [1, 2],
        "local_positions": [11, 22],  # Should never be those values, but here for testing purposes
        "patches": [PPAD, PPAD],
        "patch_positions": [POS_PAD, POS_PAD],
        "input_types": [0, 0],
        "loss_mask": [0, 0],
    }
    local_positions = [33, 44]

    interleaver._dict_append(batch_data, processed_data, local_positions, loss_mask_value=1)

    expected_output = {
        "input_ids": [1, 2, 101, 102],
        "local_positions": [11, 22, 33, 44],
        "patches": [PPAD, PPAD, PPAD, PPAD],
        "patch_positions": [POS_PAD, POS_PAD, POS_PAD, POS_PAD],
        "input_types": [0, 0, 0, 0],
        "loss_mask": [0, 0, 1, 1],
    }

    # To make dicts easier to compare:
    processed_data["patches"] = [p.tolist() for p in processed_data["patches"]]
    expected_output["patches"] = [p.tolist() for p in expected_output["patches"]]

    assert processed_data == expected_output


def test_dict_append_patches_and_patch_positions():
    interleaver = Interleaver()
    batch_data = {"patches": [P1, P2], "patch_positions": [LEFT, RIGHT]}
    processed_data = {
        "input_ids": [1, 2],
        "local_positions": [11, 22],  # Should never be those values, but here for testing purposes
        "patches": [PPAD, PPAD],
        "patch_positions": [POS_PAD, POS_PAD],
        "input_types": [0, 0],
        "loss_mask": [0, 0],
    }
    local_positions = [33, 44]

    interleaver._dict_append(batch_data, processed_data, local_positions, loss_mask_value=1)

    expected_output = {
        "input_ids": [1, 2, 0, 0],
        "local_positions": [11, 22, 33, 44],
        "patches": [PPAD, PPAD, P1, P2],
        "patch_positions": [POS_PAD, POS_PAD, LEFT, RIGHT],
        "input_types": [0, 0, 1, 1],
        "loss_mask": [0, 0, 1, 1],
    }

    # To make dicts easier to compare:
    processed_data["patches"] = [p.tolist() for p in processed_data["patches"]]
    expected_output["patches"] = [p.tolist() for p in expected_output["patches"]]

    assert processed_data == expected_output


def test_interleave_episode():
    # 3 timesteps
    # 1 observation is composed of
    #     2 image patches (with the associated patch_positions) and
    #     1 discrete observations composed of 3 ints
    st = 42
    lppv = -1
    interleaver = Interleaver(separator_token=st, local_position_pad_value=lppv)
    sample_data = {
        "discrete_observations": {
            "input_ids": [[1, 2], [3, 4]],
        },
        "image_observations": {
            "patches": [[P1, P2], [P3, P4]],
            "patch_positions": [[LEFT, RIGHT], [LEFT, RIGHT]],
        },
    }
    expected_output = {
        "input_ids": [0, 0, 1, 2, st, 0, 0, 3, 4, st],
        "local_positions": [0, 1, 2, 3, lppv, 0, 1, 2, 3, lppv],
        "patches": [P1, P2, PPAD, PPAD, PPAD, P3, P4, PPAD, PPAD, PPAD],
        "patch_positions": [LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD],
        "input_types": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        "loss_mask": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    }

    processed_data = interleaver._interleave_episode(sample_data)

    # To make dicts easier to compare:
    processed_data["patches"] = [p.tolist() for p in processed_data["patches"]]
    expected_output["patches"] = [p.tolist() for p in expected_output["patches"]]

    assert processed_data == expected_output


def test_interleave_standalone():
    interleaver = Interleaver()
    input_data = {
        "text": {"input_ids": [1, 2]},
        "images": {"patches": [P1, P2], "patch_positions": [LEFT, RIGHT]},
    }
    expected_output = {
        "input_ids": [0, 0, 1, 2],
        "local_positions": [-1, -1, -1, -1],
        "patches": [P1, P2, PPAD, PPAD],
        "patch_positions": [LEFT, RIGHT, POS_PAD, POS_PAD],
        "input_types": [1, 1, 0, 0],
        "loss_mask": [0, 0, 1, 1],
    }

    processed_data = interleaver._interleave_standalone(input_data)

    # To make dicts easier to compare:
    processed_data["patches"] = [p.tolist() for p in processed_data["patches"]]
    expected_output["patches"] = [p.tolist() for p in expected_output["patches"]]

    assert processed_data == expected_output


def test_is_episode():
    # Test valid non-episode case
    sample_data_1 = {
        "images": {"patches": [], "patch_positions": []},
        "text": {"input_ids": []},
    }
    assert not Interleaver._is_episode(sample_data_1)

    # Test valid episode case
    sample_data_2 = {
        "image_observations": {"patches": [], "patch_positions": []},
        "text_observations": {"input_ids": []},
        "discrete_observations": {"input_ids": []},
        "continuous_actions": {"input_ids": []},
    }
    assert Interleaver._is_episode(sample_data_2)

    # Test invalid mixed case
    sample_data_3 = {
        "images": {"patches": [], "patch_positions": []},
        "text_observations": {"input_ids": []},
        "continuous_actions": {"input_ids": []},
    }
    with pytest.raises(ValueError):
        Interleaver._is_episode(sample_data_3)

    # Test case with extra keys
    sample_data_4 = {
        "images": {"patches": [], "patch_positions": []},
        "text": {"input_ids": []},
        "extra_key": {"input_ids": []},
    }
    with pytest.raises(ValueError):
        Interleaver._is_episode(sample_data_4)


def test_interleave_batch_episode():
    st = 42
    lppv = -1
    interleaver = Interleaver(separator_token=st, local_position_pad_value=lppv)
    input_data = {
        "text_observations": {
            "input_ids": [
                None,
                [[1, 2], [3, 4], [5, 6]],
            ]
        },
        "image_observations": {
            "patches": [
                [[P1, P2], [P3, P4]],
                None,
            ],
            "patch_positions": [
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
            [0, 0, st, 11, 12, 0, 0, st, 13, 14],
            [1, 2, st, 15, 16, 3, 4, st, 17, 18, 5, 6, st, 19, 20],
        ],
        "local_positions": [
            [0, 1, lppv, lppv, lppv, 0, 1, lppv, lppv, lppv],
            [0, 1, lppv, lppv, lppv, 0, 1, lppv, lppv, lppv, 0, 1, lppv, lppv, lppv],
        ],
        "patches": [
            [P1, P2, PPAD, PPAD, PPAD, P3, P4, PPAD, PPAD, PPAD],
            [PPAD] * 15,
        ],
        "patch_positions": [
            [LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD],
            [POS_PAD] * 15,
        ],
        "input_types": [
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "loss_mask": [
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
    }
    processed_data = interleaver(input_data)

    # To make dicts easier to compare:
    processed_data["patches"] = [[p.tolist() for p in ep] for ep in processed_data["patches"]]
    expected_output["patches"] = [[p.tolist() for p in ep] for ep in expected_output["patches"]]

    assert processed_data == expected_output


def test_interleave_batch_standalone():
    st = 42
    lppv = -1
    interleaver = Interleaver(separator_token=st, local_position_pad_value=lppv)
    input_data = {
        "text": {
            "input_ids": [None, [1, 2], [3, 4]],
        },
        "images": {
            "patches": [[P1, P2], None, [P3, P4]],
            "patch_positions": [[LEFT, RIGHT], None, [LEFT, RIGHT]],
        },
    }
    expected_output = {
        "input_ids": [
            [0, 0],
            [1, 2],
            [0, 0, 3, 4],
        ],
        "local_positions": [
            [lppv, lppv],
            [lppv, lppv],
            [lppv, lppv, lppv, lppv],
        ],
        "patches": [
            [P1, P2],
            [PPAD, PPAD],
            [P3, P4, PPAD, PPAD],
        ],
        "patch_positions": [
            [LEFT, RIGHT],
            [POS_PAD, POS_PAD],
            [LEFT, RIGHT, POS_PAD, POS_PAD],
        ],
        "input_types": [
            [1, 1],
            [0, 0],
            [1, 1, 0, 0],
        ],
        "loss_mask": [[0, 0], [1, 1], [0, 0, 1, 1]],
    }

    processed_data = interleaver(input_data)

    # To make dicts easier to compare:
    processed_data["patches"] = [[p.tolist() for p in ep] for ep in processed_data["patches"]]
    expected_output["patches"] = [[p.tolist() for p in ep] for ep in expected_output["patches"]]

    assert processed_data == expected_output


def test_interleave_batch_mixed():
    st = 42
    lppv = -1
    interleaver = Interleaver(separator_token=st, local_position_pad_value=lppv)
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
                [[P1, P2], [P3, P4]],
            ],
            "patch_positions": [
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
            [0, 0, st, 11, 12, 0, 0, st, 13, 14],
        ],
        "local_positions": [
            [-1, -1],
            [0, 1, lppv, lppv, lppv, 0, 1, lppv, lppv, lppv],
        ],
        "patches": [
            [PPAD, PPAD],
            [P1, P2, PPAD, PPAD, PPAD, P3, P4, PPAD, PPAD, PPAD],
        ],
        "patch_positions": [
            [POS_PAD, POS_PAD],
            [LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD, LEFT, RIGHT, POS_PAD, POS_PAD, POS_PAD],
        ],
        "input_types": [
            [0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        ],
        "loss_mask": [
            [1, 1],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        ],
    }

    processed_data = interleaver(input_data)

    # To make dicts easier to compare:
    processed_data["patches"] = [[p.tolist() for p in ep] for ep in processed_data["patches"]]
    expected_output["patches"] = [[p.tolist() for p in ep] for ep in expected_output["patches"]]

    assert processed_data == expected_output
