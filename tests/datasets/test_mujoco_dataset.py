import pytest
import numpy as np
from gia.datasets import MujocoDataset, MujocoTaskDataset
from gia.datasets.utils import mu_law_np


def test_mujoco_pack():
    # tests the packing of episodes of obs, done and actions into tokenized sequences

    obs_eps = [
        np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        ),
        np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        ),
    ]
    act_eps = [
        np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        ),
        np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        ),
    ]
    expected_tokens = np.array(
        [
            [512, 744, 512, 744, 779, 799, 779, 799, 0],
            [814, 825, 814, 825, 512, 744, 512, 744, 0],
        ]
    )
    expected_attn = np.array(
        [
            [[0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 0]],
            [[0, 3], [0, 3], [0, 3], [0, 3], [4, 7], [4, 7], [4, 7], [4, 7], [0, 0]],
        ]
    )
    expected_positions = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
        ]
    )

    packed_tokens, packed_attn, packed_positions = MujocoTaskDataset.pack(obs_eps, act_eps, seq_len=9)

    assert packed_tokens.shape == packed_attn.shape[:-1] == packed_positions.shape
    assert np.all(packed_tokens == expected_tokens)
    assert np.all(packed_positions == expected_positions)
    assert np.all(packed_attn == expected_attn)


if __name__ == "__main__":
    test_pack()
