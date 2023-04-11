import pytest
import torch

from gia.config import Arguments
from gia.model.embedding import (
    Embeddings,
    ImageEncoder,
    ImagePositionEncoding,
    LocalPositionEncodings,
)


def random_positions(size):  # Ensure that min < max
    t1 = torch.rand(*size, 1, 2)  # Create a random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t2 = torch.rand(*size, 1, 2)  # Create another random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t_min = torch.min(t1, t2)  # Element-wise minimum of t1 and t2
    t_max = torch.max(t1, t2)  # Element-wise maximum of t1 and t2
    return torch.cat((t_min, t_max), dim=-2)  # Concatenate t_min and t_max along the proper dimension


def test_image_position_encoding_shapes():
    batch_size = 4
    positions = torch.tensor(
        [
            [[0.0, 0.0], [0.2, 0.3]],
            [[0.1, 0.3], [0.2, 0.4]],
            [[0.2, 0.4], [0.3, 0.5]],
            [[0.3, 0.5], [0.4, 0.6]],
        ]
    )

    pos_enc = ImagePositionEncoding(embed_dim=128)
    pos_encoding = pos_enc(positions)
    assert pos_encoding.shape == (batch_size, 128)

    pos_encoding_eval = pos_enc(positions, eval=True)
    assert pos_encoding_eval.shape == (batch_size, 128)


def test_image_position_encoding_values():
    # The two patches should have the same encoding since they are under 1/vocab_size
    positions = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.2]],
            [[0.0, 0.1], [0.2, 0.2]],
        ]
    )
    pos_enc = ImagePositionEncoding(vocab_size=4)
    pos_encoding = pos_enc(positions)
    assert torch.allclose(pos_encoding[0], pos_encoding[1])


def test_image_position_encoding_values_eval():
    # The two patches should have the same encoding since they share the same mean position (0.2, 0.3)
    positions = torch.tensor(
        [
            [[0.1, 0.0], [0.3, 0.6]],
            [[0.0, 0.2], [0.4, 0.4]],
        ]
    )
    pos_enc = ImagePositionEncoding(vocab_size=10)
    pos_encoding = pos_enc(positions, eval=True)
    assert torch.allclose(pos_encoding[0], pos_encoding[1])


def test_local_position_encodings():
    batch_size, seq_len, num_tokens = 8, 16, 20
    vocab_size, embed_dim = 128, 2048
    pos_enc = LocalPositionEncodings(vocab_size, embed_dim)
    shape = torch.Size([batch_size, seq_len, num_tokens, embed_dim])

    # Test when same is False
    pos_emb = pos_enc(shape)
    assert pos_emb.shape == shape, f"Expected shape {shape}, got {pos_emb.shape}"
    assert torch.allclose(pos_emb[1:], pos_emb[-1:]), "Position encodings should not depend on batch index"
    assert torch.allclose(pos_emb[:, 1:], pos_emb[:, -1:]), "Position encodings should not depend on timestep"
    assert not torch.allclose(pos_emb[:, :, 1:], pos_emb[:, :, -1:]), "Position encodings should vary locally"


def test_local_position_encodings_same():
    batch_size, seq_len, num_tokens = 8, 16, 20
    vocab_size, embed_dim = 128, 2048
    pos_enc = LocalPositionEncodings(vocab_size, embed_dim)
    shape = torch.Size([batch_size, seq_len, num_tokens, embed_dim])

    # Test when same is False
    pos_emb = pos_enc(shape, same=True)
    assert pos_emb.shape == shape, f"Expected shape {shape}, got {pos_emb.shape}"
    assert torch.allclose(pos_emb[1:], pos_emb[-1:]), "Position encodings should not depend on batch index"
    assert torch.allclose(pos_emb[:, 1:], pos_emb[:, -1:]), "Position encodings should not depend on timestep"
    assert torch.allclose(pos_emb[:, :, 1:], pos_emb[:, :, -1:]), "Position encodings should not vary locally"


def test_image_encoder():
    batch_size = 8
    in_channels = 4
    num_res_channels = 32
    out_features = 128
    num_groups = 4
    patch_size = 16

    image_encoder = ImageEncoder(in_channels, num_res_channels, out_features, num_groups, patch_size)

    # Test with input images of shape (batch_size, 3, patch_size, patch_size)
    input_images = torch.randn(batch_size, 3, patch_size, patch_size)
    encoded_images = image_encoder(input_images)
    assert encoded_images.shape == (
        batch_size,
        out_features,
    ), f"Expected shape ({batch_size}, {out_features}), got {encoded_images.shape}"

    # Test with input images of shape (batch_size, 4, patch_size, patch_size)
    input_images = torch.randn(batch_size, 4, patch_size, patch_size)
    encoded_images = image_encoder(input_images)
    assert encoded_images.shape == (
        batch_size,
        out_features,
    ), f"Expected shape ({batch_size}, {out_features}), got {encoded_images.shape}"


@pytest.mark.parametrize("obs_modality", ["discrete", "continuous", "text"])
@pytest.mark.parametrize("act_modality", ["discrete", "continuous", "text"])
@pytest.mark.parametrize("use_seprator", [True, False])
def test_embeddings(obs_modality, act_modality, use_seprator):
    batch_size, seq_len = 8, 4
    num_obs_tokens = 4
    num_act_tokens = 3
    obs_min, obs_max = (0, 32_000) if obs_modality == "text" else (32_000, 32_010)
    act_min, act_max = (0, 32_000) if act_modality == "text" else (32_000, 32_010)
    obs_shape = (batch_size, seq_len, num_obs_tokens)
    act_shape = (batch_size, seq_len, num_act_tokens)
    batch = {
        f"{obs_modality}_observations": torch.randint(obs_min, obs_max, obs_shape),
        f"{obs_modality}_observations_loss_mask": torch.randint(0, 2, obs_shape).bool(),
        f"{obs_modality}_observations_attention_mask": torch.randint(0, 2, obs_shape).bool(),
        f"{act_modality}_actions": torch.randint(act_min, act_max, act_shape),
        f"{act_modality}_actions_loss_mask": torch.randint(0, 2, act_shape).bool(),
        f"{act_modality}_actions_attention_mask": torch.randint(0, 2, act_shape).bool(),
    }
    args = Arguments(embed_dim=32, use_separator=use_seprator)
    embed = Embeddings(args)
    embeddings = embed(batch)
    # observations and actions are concatenated
    num_tokens = num_obs_tokens + num_act_tokens
    if use_seprator:
        num_tokens += 1
    expected_shape = (batch_size, seq_len * num_tokens)
    assert embeddings["tokens"].shape == expected_shape
    assert embeddings["attention_mask"].shape == expected_shape
    assert embeddings["loss_mask"].shape == expected_shape
    assert embeddings["embeddings"].shape == (*expected_shape, 32)


@pytest.mark.parametrize("act_modality", ["discrete", "continuous", "text"])
@pytest.mark.parametrize("use_seprator", [True, False])
def test_embeddings_image(act_modality, use_seprator):
    batch_size, seq_len = 8, 4
    num_patches, patch_size = 20, 16
    num_act_tokens = 3
    act_min, act_max = (0, 32_000) if act_modality == "text" else (32_000, 32_010)
    obs_shape = (batch_size, seq_len, num_patches)
    act_shape = (batch_size, seq_len, num_act_tokens)

    batch = {
        "image_observations": torch.randint(0, 255, (*obs_shape, 3, patch_size, patch_size), dtype=torch.uint8),
        "image_observations_loss_mask": torch.randint(0, 2, obs_shape).bool(),
        "image_observations_attention_mask": torch.randint(0, 2, obs_shape).bool(),
        "patches_positions": random_positions(obs_shape),
        f"{act_modality}_actions": torch.randint(act_min, act_max, act_shape),
        f"{act_modality}_actions_loss_mask": torch.randint(0, 2, act_shape).bool(),
        f"{act_modality}_actions_attention_mask": torch.randint(0, 2, act_shape).bool(),
    }
    args = Arguments(embed_dim=32, use_separator=use_seprator)
    embed = Embeddings(args)
    embeddings = embed(batch)
    # observations and actions are concatenated
    num_tokens = num_patches + num_act_tokens
    if use_seprator:
        num_tokens += 1
    expected_shape = (batch_size, seq_len * num_tokens)
    assert embeddings["tokens"].shape == expected_shape
    assert embeddings["attention_mask"].shape == expected_shape
    assert embeddings["loss_mask"].shape == expected_shape
    assert embeddings["embeddings"].shape == (*expected_shape, 32)
