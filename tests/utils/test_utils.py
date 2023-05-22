import torch
from transformers.testing_utils import CaptureLogger

from gia.utils.utils import cache_decorator, logger


@cache_decorator
def dummy_function(x: int) -> int:
    return x * x


def test_cache_decorator_cache_hit(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    result = dummy_function(x, load_from_cache=False, cache_dir=str(tmp_path))
    torch.testing.assert_close(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(x, load_from_cache=True, cache_dir=str(tmp_path))

    torch.testing.assert_close(result, 4 * torch.ones(1))
    assert "Loading cache" in cl.out
    assert str(tmp_path) in cl.out


def test_cache_decorator_cache_miss(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    y = 3 * torch.ones(1)
    result = dummy_function(x, load_from_cache=False, cache_dir=str(tmp_path))
    torch.testing.assert_close(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(y, load_from_cache=True, cache_dir=str(tmp_path))

    torch.testing.assert_close(result, 9 * torch.ones(1))
    assert "Loading cache" not in cl.out


def test_cache_decorator_cache_hit_no_cache(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    result = dummy_function(x, load_from_cache=False, cache_dir=str(tmp_path))
    torch.testing.assert_close(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(x, load_from_cache=False, cache_dir=str(tmp_path))

    torch.testing.assert_close(result, 4 * torch.ones(1))
    assert "Loading cache" not in cl.out


def test_cache_decorator_cache_miss_no_cache(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    y = 3 * torch.ones(1)
    result = dummy_function(x, load_from_cache=False, cache_dir=str(tmp_path))
    torch.testing.assert_close(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(y, load_from_cache=True, cache_dir=str(tmp_path))

    torch.testing.assert_close(result, 9 * torch.ones(1))
    assert "Loading cache" not in cl.out
