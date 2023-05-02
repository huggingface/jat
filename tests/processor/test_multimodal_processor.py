import numpy as np

from gia.config import DatasetArguments
from gia.processing import GiaProcessor


def test_tokenize_continuous():
    # Instantiate GiaProcessor
    args = DatasetArguments(mu=10, M=25, nb_bins=1024)
    processor = GiaProcessor(args)

    # Create sample tokens
    x = np.linspace(-5.0, 5.0, 51).reshape(1, -1)  # Convert to batch of size 1

    # Call tokenize_continuous method
    tokens = processor.tokenize_continuous(x)

    # Compute the expected result
    expected_result = processor.inverse_tokenize_continuous(tokens)

    # Assert that the result is close to the expected result
    np.testing.assert_allclose(x, expected_result, atol=1e-1)


def test_continuous_processing():
    args = DatasetArguments(nb_bins=16)
    processor = GiaProcessor(args)
    seq_len = 10
    obs_size = 9
    action_size = 3
    # Generate 1 episode of random observations and actions
    observations = [np.random.rand(seq_len, obs_size) * 20 - 10]  # random observations in [-10, 10]
    actions = [np.random.rand(seq_len, action_size) * 20 - 10]  # random actions in [-10, 10]
    tokens = processor(continuous_observations=observations, continuous_actions=actions)
    # Check that we've still one episode
    assert len(tokens["continuous_observations"]) == 1
    assert len(tokens["continuous_actions"]) == 1
    # Check the the sequence length is correct
    assert len(tokens["continuous_observations"][0]) == seq_len
    assert len(tokens["continuous_actions"][0]) == seq_len
    # Check that the shape is correct
    assert tokens["continuous_observations"][0].shape == (seq_len, obs_size)
    assert tokens["continuous_actions"][0].shape == (seq_len, action_size)
    # Check that the dtype is correct
    assert tokens["continuous_observations"][0].dtype == np.int64
    assert tokens["continuous_actions"][0].dtype == np.int64
    # Check that the tokens are shifted (caution, tokens shouldn't be the separator tokens)
    assert np.all(tokens["continuous_observations"][0] >= processor.token_shift)  # Tokens are shifted
    assert np.all(tokens["continuous_actions"][0] >= processor.token_shift)  # Tokens are shifted
    assert np.all(tokens["continuous_observations"][0] <= args.nb_bins + processor.token_shift)
    assert np.all(tokens["continuous_actions"][0] <= args.nb_bins + processor.token_shift)


def test_discrete_processing():
    args = DatasetArguments(nb_bins=16)
    processor = GiaProcessor(args)
    seq_len = 10
    obs_size = 9
    action_size = 3
    # Generate 1 episode of random observations and actions
    observations = [np.random.randint(0, 5, (seq_len, obs_size))]  # random observations in [0, 5]
    actions = [np.random.randint(0, 4, (seq_len, action_size))]  # random actions in [0, 4]
    tokens = processor(discrete_observations=observations, discrete_actions=actions)
    # Check that we've still one episode
    assert len(tokens["discrete_observations"]) == 1
    assert len(tokens["discrete_actions"]) == 1
    # Check the the sequence length is correct
    assert len(tokens["discrete_observations"][0]) == seq_len
    assert len(tokens["discrete_actions"][0]) == seq_len
    # Check that the shape is correct
    assert tokens["discrete_observations"][0].shape == (seq_len, obs_size)
    assert tokens["discrete_actions"][0].shape == (seq_len, action_size)
    # Check that the dtype is correct
    assert tokens["discrete_observations"][0].dtype == np.int64
    assert tokens["discrete_actions"][0].dtype == np.int64
    # Check that the tokens are shifted (caution, tokens shouldn't be the separator tokens)
    assert np.all(tokens["discrete_observations"][0] >= processor.token_shift)  # Tokens are shifted
    assert np.all(tokens["discrete_actions"][0] >= processor.token_shift)  # Tokens are shifted
    assert np.all(tokens["discrete_observations"][0] <= args.nb_bins + processor.token_shift)
    assert np.all(tokens["discrete_actions"][0] <= args.nb_bins + processor.token_shift)


def test_text_only_processing():
    # In text-only dataset, we don't have episodes
    # The model should handle this case
    args = DatasetArguments(nb_bins=16)
    processor = GiaProcessor(args)
    texts = ["This is a test", "This is another test"]
    tokens = processor(text_observations=texts)
    # Check that the returned object is a dict with the correct keys
    assert isinstance(tokens, dict)
    assert "text_observations" in tokens
    # Check that we have the correct number of samples
    assert len(tokens["text_observations"]) == len(texts)
    # Check that the dtype is correct
    assert tokens["text_observations"][0].dtype == np.int64
    assert tokens["text_observations"][1].dtype == np.int64
    # The two texts are different, so the tokens should be different
    assert np.any(tokens["text_observations"][0] != tokens["text_observations"][1])
    # Check the token values
    assert np.all(tokens["text_observations"][0] < processor.token_shift)  # Tokens aren't shifted
    assert np.all(tokens["text_observations"][1] < processor.token_shift)  # Tokens aren't shifted


def test_mixed_batch_processing():
    args = DatasetArguments(nb_bins=16)
    processor = GiaProcessor(args)
    continuous_observations = [
        [[0.1, 0.2], [0.3, 0.4]],
        None,
    ]
    discrete_observations = [
        None,
        [[1, 2], [3, 4]],
    ]
    continuous_actions = [
        [[0.5, 0.6], [0.7, 0.8]],
        [[0.9, 1.0], [1.1, 1.2]],
    ]

    tokens = processor(
        continuous_observations=continuous_observations,
        discrete_observations=discrete_observations,
        continuous_actions=continuous_actions,
    )
    # Check that we've still two episodes
    assert len(tokens["continuous_observations"]) == 2
    assert len(tokens["discrete_observations"]) == 2
    assert len(tokens["continuous_actions"]) == 2
    # Check that there are still None values
    assert tokens["continuous_observations"][1] is None
    assert tokens["discrete_observations"][0] is None
    # Check the the sequence length is correct
    assert len(tokens["continuous_observations"][0]) == 2
    assert len(tokens["discrete_observations"][1]) == 2
    assert len(tokens["continuous_actions"][0]) == 2
    assert len(tokens["continuous_actions"][1]) == 2
    # TODO: finish this test
