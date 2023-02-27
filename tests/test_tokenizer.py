import torch

from gia.tokenizers import Tokenizer


def test_continuous_tokenizer():
    continous_tokenizer = Tokenizer()
    observations = torch.rand(10, 9) * 20 - 10  # Random observations in [-10, 10]
    actions = torch.rand(10, 3) * 20 - 10  # Random actions in [-10, 10]
    tokens = continous_tokenizer(observations, actions)
    assert tokens.shape == (10, 13)  # 9 observations, 1 separator, 3 actions
    assert torch.all(tokens[:, :9] != continous_tokenizer.nb_bins)  # Observation tokens are not separator tokens
    assert torch.all(tokens[:, 9] == continous_tokenizer.nb_bins)  # Separator token
    assert torch.all(tokens[:, 10:] != continous_tokenizer.nb_bins)  # Action tokens are not separator tokens
