import numpy as np

from gia.datasets.gia_dataset import is_continuous, is_discrete, is_image, is_text


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
