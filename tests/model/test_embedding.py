import torch

from gia.model.embedding import Embeddings


def test_embeddings():
    batch_size, seq_len, num_tokens = 8, 4, 3
    batch = {
        "discrete_observations": torch.randint(0, 32_000, (batch_size, seq_len, num_tokens)),
        "discrete_observations_loss_mask": torch.randint(0, 2, (batch_size, seq_len, num_tokens)).bool(),
        "discrete_observations_attention_mask": torch.randint(0, 2, (batch_size, seq_len, num_tokens)).bool(),
        "discrete_actions": torch.randint(32_000, 32_010, (batch_size, seq_len, num_tokens)),
        "discrete_actions_loss_mask": torch.randint(32_000, 32_010, (batch_size, seq_len, num_tokens)).bool(),
        "discrete_actions_attention_mask": torch.randint(32_000, 32_010, (batch_size, seq_len, num_tokens)).bool(),
    }
    embed = Embeddings(embedding_dim=32)
    embeddings = embed(batch)
    # observations and actions are concatenated
    assert embeddings["tokens"].shape == (batch_size, seq_len * num_tokens * 2)
    assert embeddings["attention_mask"].shape == (batch_size, seq_len * num_tokens * 2)
    assert embeddings["loss_mask"].shape == (batch_size, seq_len * num_tokens * 2)
    assert embeddings["embeddings"].shape == (batch_size, seq_len * num_tokens * 2, 32)
