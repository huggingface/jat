import pytest
import torch
from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import collate_fn, load_gia_dataset
from gia.datasets.utils import DatasetDict
from gia.model.embedding import Embeddings, ImageEncoder, ImagePositionEncoding, LocalPositionEncodings
from gia.processing import GiaProcessor


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
    torch.testing.assert_close(pos_encoding[0], pos_encoding[1])


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
    torch.testing.assert_close(pos_encoding[0], pos_encoding[1])


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


def test_local_position_encodings():
    batch_size, seq_len, num_tokens = 8, 16, 20
    vocab_size, embed_dim = 128, 2048
    pos_enc = LocalPositionEncodings(vocab_size, embed_dim)
    shape = torch.Size([batch_size, seq_len, num_tokens, embed_dim])

    # Test when same is False
    pos_emb = pos_enc(shape)
    assert pos_emb.shape == shape, f"Expected shape {shape}, got {pos_emb.shape}"
    torch.testing.assert_close(pos_emb[1:], pos_emb[:-1], msg="Position encodings should not depend on batch index")
    torch.testing.assert_close(pos_emb[:, 1:], pos_emb[:, :-1], msg="Position encodings should not depend on timestep")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(pos_emb[:, :, 1:], pos_emb[:, :, :-1], msg="Position encodings should vary locally")


def test_local_position_encodings_same():
    batch_size, seq_len, num_tokens = 8, 16, 20
    vocab_size, embed_dim = 128, 2048
    pos_enc = LocalPositionEncodings(vocab_size, embed_dim)
    shape = torch.Size([batch_size, seq_len, num_tokens, embed_dim])

    # Test when same is False
    pos_emb = pos_enc(shape, same=True)
    assert pos_emb.shape == shape, f"Expected shape {shape}, got {pos_emb.shape}"
    torch.testing.assert_close(pos_emb[1:], pos_emb[:-1], msg="Position encodings should not depend on batch index")
    torch.testing.assert_close(pos_emb[:, 1:], pos_emb[:, :-1], msg="Position encodings should not depend on timestep")
    torch.testing.assert_close(pos_emb[:, :, 1:], pos_emb[:, :, :-1], msg="Position encodings should not vary locally")


def test_embeddings():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 32))
    patches = torch.rand(2, 32, 4, 16, 16)
    positions = random_positions((2, 32))
    input_type = torch.randint(0, 2, (2, 32))
    attention_mask = torch.randint(0, 2, (2, 32), dtype=torch.bool)
    output_tensor = module(input_ids, patches, positions, input_type, attention_mask)
    assert output_tensor.shape == (2, 32, 128)


def test_embed_real_data():
    args = Arguments(task_names=["mujoco-ant"], embed_dim=128, output_dir="tests/test_embed_real_data")
    dataset = load_gia_dataset(args.task_names)
    processor = GiaProcessor(args)
    dataset = DatasetDict(processor(**dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    embeddings = Embeddings(embed_dim=args.embed_dim, token_vocab_size=32000 + 1024)
    batch = next(iter(dataloader))
    batch.pop("loss_mask")  # not an arg of embeddings
    embeds = embeddings(**batch)
    assert embeds.shape == (args.batch_size, args.seq_len, args.embed_dim)
