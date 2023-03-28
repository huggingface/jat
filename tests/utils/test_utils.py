import numpy as np
import pytest
import torch
from transformers.testing_utils import CaptureLogger

from gia.utils.utils import cache_decorator, discretize, inverse_mu_law, logger, mu_law


def test_mu_law_positive_input():
    x = np.array([0.5, 0.75, 1.0])
    y = mu_law(x)
    expected_output = np.array([0.3874, 0.4267, 0.4547])
    assert np.allclose(y, expected_output, atol=1e-04)


def test_mu_law_negative_input():
    x = np.array([-0.5, -0.75, -1.0])
    y = mu_law(x)
    expected_output = np.array([-0.3874, -0.4267, -0.4547])
    assert np.allclose(y, expected_output, atol=1e-04)


def test_mu_law_zero_input():
    x = np.array([0.0])
    y = mu_law(x)
    expected_output = np.array([0.0])
    assert np.allclose(y, expected_output, atol=1e-04)


def test_mu_law_custom_mu():
    x = np.array([0.5, 0.75, 1.0])
    y = mu_law(x, mu=50)
    expected_output = np.array([0.3445, 0.3860, 0.4157])
    assert np.allclose(y, expected_output, atol=1e-04)


def test_mu_law_custom_M():
    x = np.array([0.5, 0.75, 1.0])
    y = mu_law(x, M=512)
    expected_output = np.array([0.3626, 0.3994, 0.4256])
    assert np.allclose(y, expected_output, atol=1e-04)


def test_discretize():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = discretize(x)
    expected_output = np.array([0, 256, 512, 768, 1023])
    assert np.allclose(y, expected_output)


def test_custom_nb_bins():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = discretize(x, nb_bins=512)
    expected_output = np.array([0, 128, 256, 384, 511])
    assert np.allclose(y, expected_output)


def test_input_bounds():
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    with pytest.raises(ValueError):
        discretize(x)


def test_inverse_mu_law():
    x = np.linspace(-5.0, 5.0, 50)
    y = mu_law(x)
    z = inverse_mu_law(y)
    assert np.allclose(x, z, atol=1e-04)


def test_inverse_mu_law2():
    x = np.linspace(-1.0, 1.0, 50)
    y = inverse_mu_law(x)
    z = mu_law(y)
    assert np.allclose(x, z, atol=1e-04)


@cache_decorator
def dummy_function(x: int) -> int:
    return x * x


def test_cache_decorator_cache_hit(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    result = dummy_function(x, load_from_cache_file=False, cache_dir=str(tmp_path))
    torch.testing.assert_allclose(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(x, load_from_cache_file=True, cache_dir=str(tmp_path))

    torch.testing.assert_allclose(result, 4 * torch.ones(1))
    assert "Loading cache" in cl.out
    assert str(tmp_path) in cl.out


def test_cache_decorator_cache_miss(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    y = 3 * torch.ones(1)
    result = dummy_function(x, load_from_cache_file=False, cache_dir=str(tmp_path))
    torch.testing.assert_allclose(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(y, load_from_cache_file=True, cache_dir=str(tmp_path))

    torch.testing.assert_allclose(result, 9 * torch.ones(1))
    assert "Loading cache" not in cl.out


def test_cache_decorator_cache_hit_no_cache(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    result = dummy_function(x, load_from_cache_file=False, cache_dir=str(tmp_path))
    torch.testing.assert_allclose(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(x, load_from_cache_file=False, cache_dir=str(tmp_path))

    torch.testing.assert_allclose(result, 4 * torch.ones(1))
    assert "Loading cache" not in cl.out


def test_cache_decorator_cache_miss_no_cache(tmp_path):
    # First call to create the cache file
    x = 2 * torch.ones(1)
    y = 3 * torch.ones(1)
    result = dummy_function(x, load_from_cache_file=False, cache_dir=str(tmp_path))
    torch.testing.assert_allclose(result, 4 * torch.ones(1))

    # Second call to load from the cache file
    with CaptureLogger(logger) as cl:
        result = dummy_function(y, load_from_cache_file=True, cache_dir=str(tmp_path))

    torch.testing.assert_allclose(result, 9 * torch.ones(1))
    assert "Loading cache" not in cl.out
