import numpy as np

from gia.datasets import MujocoTaskDataset


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
            [[0, 2], [0, 2], [0, 2], [0, 2], [0, 6], [0, 6], [0, 6], [0, 6], [0, 0]],
            [[0, 2], [0, 2], [0, 2], [0, 2], [4, 6], [4, 6], [4, 6], [4, 6], [0, 0]],
        ]
    )
    expected_positions = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
        ]
    )
    expected_loss_mask = np.array(
        [
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
        ]
    )

    packed_tokens, packed_attn, packed_positions, loss_mask = MujocoTaskDataset.pack(obs_eps, act_eps, seq_len=9)

    assert packed_tokens.shape == packed_attn.shape[:-1] == packed_positions.shape
    assert np.all(packed_tokens == expected_tokens)
    assert np.all(packed_positions == expected_positions)
    assert np.all(packed_attn == expected_attn)
    assert np.all(loss_mask == expected_loss_mask)
