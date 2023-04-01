import numpy as np

from gia.processor.multimodal_processor import MultimodalProcessor


def test_tokenize_continuous():
    # Instantiate MultimodalProcessor
    processor = MultimodalProcessor(mu=100, M=256, nb_bins=1024)

    # Create sample tokens
    x = np.linspace(-5.0, 5.0, 51).reshape(1, -1)  # Convert to batch of size 1

    # Call tokenize_continuous method
    tokens = processor.tokenize_continuous(x)

    # Compute the expected result
    expected_result = processor.inverse_tokenize_continuous(tokens)

    # Assert that the result is close to the expected result
    np.testing.assert_allclose(x, expected_result, atol=1e-1)


def test_continuous_processing():
    nb_bins = 16
    token_shift = 256
    processor = MultimodalProcessor(nb_bins=nb_bins, token_shift=token_shift)
    observations = np.random.rand(10, 9) * 20 - 10  # Random array in [-10, 10]
    actions = np.random.rand(10, 3) * 20 - 10  # Random array in [-10, 10]
    tokens = processor({"continuous_observations": observations, "continuous_actions": actions})
    assert tokens["continuous_observations"].shape == observations.shape
    assert tokens["continuous_actions"].shape == actions.shape
    assert tokens["continuous_observations"].dtype == np.int64
    assert tokens["continuous_actions"].dtype == np.int64
    assert np.all(tokens["continuous_observations"] >= token_shift)  # Tokens are shifted
    assert np.all(tokens["continuous_actions"] >= token_shift)  # Tokens are shifted
    assert np.all(tokens["continuous_observations"] <= nb_bins + token_shift)
    assert np.all(tokens["continuous_actions"] <= nb_bins + token_shift)  # Observation tokens are not separator tokens


def test_discrete_tokenizer():
    nb_bins = 16
    token_shift = 256
    processor = MultimodalProcessor(nb_bins=nb_bins, token_shift=token_shift)
    observations = np.random.randint(0, 10, (10, 9))  # Random array in [0, 10]
    actions = np.random.randint(0, 10, (10, 3))  # Random array in [0, 10]
    tokens = processor({"continuous_observations": observations, "continuous_actions": actions})
    assert tokens["continuous_observations"].shape == observations.shape
    assert tokens["continuous_actions"].shape == actions.shape
    assert tokens["continuous_observations"].dtype == np.int64
    assert tokens["continuous_actions"].dtype == np.int64
    assert np.all(tokens["continuous_observations"] >= token_shift)  # Tokens are shifted
    assert np.all(tokens["continuous_actions"] >= token_shift)  # Tokens are shifted
    assert np.all(tokens["continuous_observations"] <= nb_bins + token_shift)
    assert np.all(tokens["continuous_actions"] <= nb_bins + token_shift)  # Observation tokens are not separator tokens
