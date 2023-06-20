from gia.eval.utils import is_slurm_available


def test_is_slurm_available():
    # hard to mock the case where it is available
    assert is_slurm_available() is False
