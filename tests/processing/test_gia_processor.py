import numpy as np

from gia.config import DatasetArguments
from gia.processing import GiaProcessor
from gia.processing.processing import GiaTokenizer


def test_inverse_tokenize_continuous():
    # Instantiate GiaProcessor
    args = DatasetArguments(mu=10, M=25, nb_bins=1024)
    tokenizer = GiaTokenizer(args)

    # Create sample tokens
    x = np.linspace(-5.0, 5.0, 50).reshape(1, 25, -1).tolist()  # Convert to batch of size 1, 25 timesteps, 2 features

    # Call tokenize_continuous method
    tokens = tokenizer(continuous_observations=x)

    # Compute the expected result
    expected_result = tokenizer.inverse_tokenize_continuous(tokens["continuous_observations"]["input_ids"])

    # Assert that the result is close to the expected result
    np.testing.assert_allclose(x, expected_result, atol=1e-1)


def test_tokenize_continuous():
    args = DatasetArguments(nb_bins=16)
    tokenizer = GiaTokenizer(args)
    seq_len = 10
    obs_size = 9
    action_size = 3
    # Generate 1 episode of random observations and actions
    observations = (np.random.rand(1, seq_len, obs_size) * 20 - 10).tolist()  # random observations in [-10, 10]
    actions = (np.random.rand(1, seq_len, action_size) * 20 - 10).tolist()  # random actions in [-10, 10]
    tokens = tokenizer(continuous_observations=observations, continuous_actions=actions)
    observations_tokens = np.array(tokens["continuous_observations"]["input_ids"])
    actions_tokens = np.array(tokens["continuous_actions"]["input_ids"])
    # Check that we've still one episode
    assert observations_tokens.shape == (1, seq_len, obs_size)
    assert actions_tokens.shape == (1, seq_len, action_size)
    # Check that the dtype is correct
    assert observations_tokens.dtype == np.int64
    assert actions_tokens.dtype == np.int64
    # Check that the tokens are shifted (caution, tokens shouldn't be the separator tokens)
    assert np.all(observations_tokens >= tokenizer.token_shift)  # Tokens are shifted
    assert np.all(actions_tokens >= tokenizer.token_shift)  # Tokens are shifted
    assert np.all(observations_tokens <= args.nb_bins + tokenizer.token_shift)
    assert np.all(actions_tokens <= args.nb_bins + tokenizer.token_shift)


def test_tokenize_discrete():
    args = DatasetArguments(nb_bins=16)
    tokenizer = GiaTokenizer(args)
    seq_len = 10
    obs_size = 9
    action_size = 3
    # Generate 1 episode of random observations and actions
    observations = np.random.randint(0, 5, (1, seq_len, obs_size)).tolist()  # random observations in [0, 5]
    actions = np.random.randint(0, 4, (1, seq_len, action_size)).tolist()  # random actions in [0, 4]
    tokens = tokenizer(discrete_observations=observations, discrete_actions=actions)
    observations_tokens = np.array(tokens["discrete_observations"]["input_ids"])
    actions_tokens = np.array(tokens["discrete_actions"]["input_ids"])
    # Check that we've still one episode
    assert observations_tokens.shape == (1, seq_len, obs_size)
    assert actions_tokens.shape == (1, seq_len, action_size)
    # Check that the dtype is correct
    assert observations_tokens.dtype == np.int64
    assert actions_tokens.dtype == np.int64
    # Check that the tokens are shifted (caution, tokens shouldn't be the separator tokens)
    assert np.all(observations_tokens >= tokenizer.token_shift)  # Tokens are shifted
    assert np.all(actions_tokens >= tokenizer.token_shift)  # Tokens are shifted
    assert np.all(observations_tokens <= args.nb_bins + tokenizer.token_shift)
    assert np.all(actions_tokens <= args.nb_bins + tokenizer.token_shift)


def test_tokenize_text():
    # In text-only dataset, we don't have episodes
    # The model should handle this case
    args = DatasetArguments(nb_bins=16)
    tokenizer = GiaTokenizer(args)
    texts = ["This is a test", "This is another test"]
    tokens = tokenizer(text_observations=texts)
    # Check that the returned object is a dict with the correct keys
    assert isinstance(tokens, dict)
    assert "text_observations" in tokens
    text_tokens = tokens["text_observations"]["input_ids"]
    # Check that we have the correct number of samples
    assert len(text_tokens) == len(texts)
    # Check that the dtype is correct
    assert all(isinstance(token, int) for tokens in text_tokens for token in tokens)
    # The two texts are different, so the tokens should be different
    assert any(t1 != t2 for t1, t2 in zip(*text_tokens))
    # Check the token values
    assert all(token < tokenizer.token_shift for tokens in text_tokens for token in tokens)  # Tokens aren't shifted


def test_tokenize_mixed_batch():
    args = DatasetArguments(nb_bins=16)
    tokenizer = GiaTokenizer(args)
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

    tokens = tokenizer(
        continuous_observations=continuous_observations,
        discrete_observations=discrete_observations,
        continuous_actions=continuous_actions,
    )
    # Check that we've still two episodes
    assert len(tokens["continuous_observations"]["input_ids"]) == 2
    assert len(tokens["discrete_observations"]["input_ids"]) == 2
    assert len(tokens["continuous_actions"]["input_ids"]) == 2
    # Check that there are still None values
    assert tokens["continuous_observations"]["input_ids"][1] is None
    assert tokens["discrete_observations"]["input_ids"][0] is None
    # Check the the sequence length is correct
    assert len(tokens["continuous_observations"]["input_ids"][0]) == 2
    assert len(tokens["discrete_observations"]["input_ids"][1]) == 2
    assert len(tokens["continuous_actions"]["input_ids"][0]) == 2
    assert len(tokens["continuous_actions"]["input_ids"][1]) == 2


def test_gia_processor():
    args = DatasetArguments(nb_bins=16)
    processor = GiaProcessor(args=args)
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
    rewards = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    out = processor(
        continuous_observations=continuous_observations,
        discrete_observations=discrete_observations,
        continuous_actions=continuous_actions,
        rewards=rewards,
    )

    assert isinstance(out, dict)
    assert set(out.keys()) == {"input_ids", "patches", "positions", "input_type", "attention_mask"}

    # Check that the number of samples is correct
    assert len(out["input_ids"]) == 2
    assert len(out["patches"]) == 2
    assert len(out["positions"]) == 2

    # Check that the sequence length is correct
    assert len(out["input_ids"][0]) == 1024
    assert sum(out["attention_mask"][0]) == 2 * 2 * 2  #  2 timesteps, 2 observations, 2 actions
