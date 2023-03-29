import torch

from gia.model.embedding import Embeddings, LocalPositionEncodings


def test_local_position_encodings():
    batch_size, seq_len, num_tokens = 8, 16, 20
    vocab_size, embedding_dim = 128, 2048
    pos_enc = LocalPositionEncodings(vocab_size, embedding_dim)
    shape = torch.Size([batch_size, seq_len, num_tokens, embedding_dim])

    # Test when same is False
    pos_emb = pos_enc(shape)
    assert pos_emb.shape == shape, f"Expected shape {shape}, got {pos_emb.shape}"
    assert torch.allclose(pos_emb[1:], pos_emb[-1:]), "Position encodings should not depend on batch index"
    assert torch.allclose(pos_emb[:, 1:], pos_emb[:, -1:]), "Position encodings should not depend on timestep"
    assert not torch.allclose(pos_emb[:, :, 1:], pos_emb[:, :, -1:]), "Position encodings should vary locally"


def test_local_position_encodings_same():
    batch_size, seq_len, num_tokens = 8, 16, 20
    vocab_size, embedding_dim = 128, 2048
    pos_enc = LocalPositionEncodings(vocab_size, embedding_dim)
    shape = torch.Size([batch_size, seq_len, num_tokens, embedding_dim])

    # Test when same is False
    pos_emb = pos_enc(shape, same=True)
    assert pos_emb.shape == shape, f"Expected shape {shape}, got {pos_emb.shape}"
    assert torch.allclose(pos_emb[1:], pos_emb[-1:]), "Position encodings should not depend on batch index"
    assert torch.allclose(pos_emb[:, 1:], pos_emb[:, -1:]), "Position encodings should not depend on timestep"
    assert torch.allclose(pos_emb[:, :, 1:], pos_emb[:, :, -1:]), "Position encodings should not vary locally"


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
