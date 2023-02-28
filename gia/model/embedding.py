import torch
from torch import Tensor, nn


class Embeddings(nn.Module):
    """
    Embedding layer.

    Example:
        >>> import torch
        >>> tokens = torch.tensor([[ 826,  170,  167, 33024,  465,  306],
        ...                        [ 275,  744,  176, 33024,  555,  890],
        ...                        [ 214,  199,  224, 33024,  162,   65]])
        >>> embedding_layer = Embeddings(embedding_dim=32)
        >>> embedding_layer(tokens).shape
        torch.Size([3, 6, 32])

    Args:
        embedding_dim (int): The embedding dimension.
        vocab_size (int, optional): The vocabulary size. Defaults to 32_000.
        nb_bins (int, optional): Number of bins. Defaults to 1024.
        max_nb_observation_tokens (int, optional): Maximum number of observation tokens. Defaults to 512.
    """

    def __init__(
        self, embedding_dim: int, vocab_size: int = 32_000, nb_bins: int = 1024, max_nb_observation_tokens: int = 512
    ):
        super().__init__()
        self.separator_token = nb_bins + vocab_size
        # The total number of tokens is the number of observation tokens + 1 for the separator token.
        self.embeddings = nn.Embedding(num_embeddings=vocab_size + nb_bins + 1, embedding_dim=embedding_dim)
        # Learnable local position encodings
        # The total number of tokens is the number of observation tokens + 1 for the unique action token
        # TODO: It's not clear whether we should add postional embeddings for the separator token.
        # For now, we don't add it.
        self.positional_emb = nn.Embedding(num_embeddings=max_nb_observation_tokens + 1, embedding_dim=embedding_dim)
        self.action_positional_emb_idx = torch.tensor(max_nb_observation_tokens, dtype=torch.long)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[0]

        # Check that the separator token is present in all sequences and that it has a constant position
        values, indices = torch.max(tokens, dim=1)
        assert torch.all(values == self.separator_token), "Separator token not found in at least one sequence."
        assert torch.all(indices[:-1] == indices[1:]), "Separator should be at the same position in all sequences."

        observation_size = torch.argmax(tokens[0])  # we assume that the separator token is always the greatest token.
        action_size = tokens.shape[1] - observation_size - 1  # -1 for the separator token

        # Embed tokens
        embeddings = self.embeddings(tokens)

        # Compute and add positional observation embeddings
        observation_pos_emb_idxs = torch.arange(observation_size).unsqueeze(0).repeat(sequence_length, 1)
        observation_pos_embeddings = self.positional_emb(observation_pos_emb_idxs)
        embeddings[:, :observation_size] += observation_pos_embeddings

        # Compute and add positional action embeddings
        action_pos_emb_idxs = self.action_positional_emb_idx.repeat(sequence_length, action_size)
        action_pos_embeddings = self.positional_emb(action_pos_emb_idxs)
        embeddings[:, observation_size + 1 :] += action_pos_embeddings
        return embeddings
