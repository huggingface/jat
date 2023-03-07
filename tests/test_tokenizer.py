import torch

from gia.tokenizers.multimodal_tokenizer import MultiModalTokenizer


def test_continuous_tokenizer():
    nb_bins = 16
    token_shift = 256
    tokenizer = MultiModalTokenizer(nb_bins=nb_bins, token_shift=token_shift)
    seprator_token = nb_bins + token_shift
    tensors = torch.rand(10, 9) * 20 - 10  # Random tensors in [-10, 10]
    actions = torch.rand(10, 3) * 20 - 10  # Random actions in [-10, 10]
    tokens = tokenizer(tensors=tensors, actions=actions)
    assert tokens.shape == (10, 13)  # 9 observations, 1 separator, 3 actions
    assert tokens.dtype == torch.int64  # Tokens are integers
    assert torch.all(tokens >= token_shift)  # Tokens are shifted
    assert torch.all(tokens[:, :9] != seprator_token)  # Observation tokens are not separator tokens
    assert torch.all(tokens[:, 9] == seprator_token)  # Separator token
    assert torch.all(tokens[:, 10:] != seprator_token)  # Action tokens are not separator tokens


def test_discrete_tokenizer():
    nb_bins = 16
    token_shift = 256
    tokenizer = MultiModalTokenizer(nb_bins=nb_bins, token_shift=token_shift)
    seprator_token = nb_bins + token_shift
    tensors = torch.randint(0, 10, (10, 9))  # Random tensors in [0, 10[
    actions = torch.randint(0, 10, (10, 3))  # Random actions in [0, 10[
    tokens = tokenizer(tensors=tensors, actions=actions)
    assert tokens.shape == (10, 13)  # 9 observations, 1 separator, 3 actions
    assert tokens.dtype == torch.int64  # Tokens are integers
    assert torch.all(tokens >= token_shift)  # Tokens are shifted
    assert torch.all(tokens[:, :9] != seprator_token)  # Observation tokens are not separator tokens
    assert torch.all(tokens[:, 9] == seprator_token)  # Separator token
    assert torch.all(tokens[:, 10:] != seprator_token)  # Action tokens are not separator tokens
