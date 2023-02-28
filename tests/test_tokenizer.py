import torch

from gia.model.tokenization import Tokenizer


def test_continuous_tokenizer():
    continous_tokenizer = Tokenizer(nb_bins=10, token_shift=256)
    seprator_token = 10 + 256
    tensors = torch.rand(10, 9) * 20 - 10  # Random tensors in [-10, 10]
    actions = torch.rand(10, 3) * 20 - 10  # Random actions in [-10, 10]
    tokens = continous_tokenizer(tensors=tensors, actions=actions)
    assert tokens.shape == (10, 13)  # 9 observations, 1 separator, 3 actions
    assert torch.all(tokens[:, :9] != seprator_token)  # Observation tokens are not separator tokens
    assert torch.all(tokens[:, 9] == seprator_token)  # Separator token
    assert torch.all(tokens[:, 10:] != seprator_token)  # Action tokens are not separator tokens
