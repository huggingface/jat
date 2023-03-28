import numpy as np
from torch.utils.data import DataLoader

from gia.datasets.gia_dataset import (
    is_continuous,
    is_discrete,
    is_image,
    is_text,
    load_gia_dataset,
)


def test_is_text():
    assert is_text(["hello", "world"])
    assert not is_text(["hello", 1])


def test_is_image():
    assert is_image(np.zeros((1, 1, 1, 1)))
    assert not is_image(np.zeros((1, 1)))
    assert not is_image(np.zeros((1, 1, 1)))


def test_is_continuous():
    assert is_continuous(np.zeros((1, 1), dtype=np.float32))
    assert is_continuous(np.zeros((1, 1), dtype=np.float64))
    assert not is_continuous(np.zeros((1, 1), dtype=np.int32))


def test_is_discrete():
    assert is_discrete(np.zeros((1, 1), dtype=np.int8))
    assert is_discrete(np.zeros((1, 1), dtype=np.uint16))
    assert not is_discrete(np.zeros((1, 1), dtype=np.float32))


def test_load_gia_dataset():
    dataset = load_gia_dataset("mujoco-ant")
    assert set(dataset.keys()) == set(
        [
            "rewards",
            "dones",
            "continuous_observations",
            "continuous_actions",
        ]
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        assert batch["continuous_observations"].shape == (2, 27)
        assert batch["continuous_actions"].shape == (2, 8)
        assert batch["rewards"].shape == (2,)
        assert batch["dones"].shape == (2,)
        break
