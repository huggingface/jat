import numpy as np

from gia.processor.multimodal_processor import MultimodalProcessor


def test_tokenize_continuous():
    # Instantiate MultimodalProcessor
    processor = MultimodalProcessor(mu=100, M=256, nb_bins=1024)

    # Create sample tokens
    x = np.linspace(-5.0, 5.0, 51)

    # Call tokenize_continuous method
    tokens = processor.tokenize_continuous(x)

    # Compute the expected result
    expected_result = processor.inverse_tokenize_continuous(tokens)

    # Assert that the result is close to the expected result
    np.testing.assert_allclose(x, expected_result, atol=1e-1)
