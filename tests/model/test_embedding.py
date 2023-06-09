import pytest
import torch

from gia.model.embedding import Embeddings, ImageEncoder, ImagePositionEncoding


def random_patch_positions(size):
    t1 = torch.rand(*size, 1, 2)  # Create a random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t2 = torch.rand(*size, 1, 2)  # Create another random tensor of shape (B, N, 1, 2) with values between 0 and 1
    t_min = torch.min(t1, t2)  # Element-wise minimum of t1 and t2
    t_max = torch.max(t1, t2)  # Element-wise maximum of t1 and t2
    return torch.cat((t_min, t_max), dim=-2)  # Concatenate t_min and t_max along the proper dimension


def test_image_position_encoding_shapes():
    batch_size = 4
    patch_positions = torch.tensor(
        [
            [[0.0, 0.0], [0.2, 0.3]],
            [[0.1, 0.3], [0.2, 0.4]],
            [[0.2, 0.4], [0.3, 0.5]],
            [[0.3, 0.5], [0.4, 0.6]],
        ]
    )

    pos_enc = ImagePositionEncoding(embed_dim=128)
    pos_enc.train()
    pos_encoding = pos_enc(patch_positions)
    assert pos_encoding.shape == (batch_size, 128)

    pos_enc.eval()
    pos_encoding_eval = pos_enc(patch_positions)
    assert pos_encoding_eval.shape == (batch_size, 128)


def test_image_position_encoding_values():
    # The two patches should have the same encoding since they are under 1/vocab_size
    patch_positions = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.2]],
            [[0.0, 0.1], [0.2, 0.2]],
        ]
    )
    pos_enc = ImagePositionEncoding(vocab_size=4)
    pos_encoding = pos_enc(patch_positions)
    torch.testing.assert_close(pos_encoding[0], pos_encoding[1])


def test_image_position_encoding_values_eval():
    # The two patches should have the same encoding since they share the same mean position (0.2, 0.3)
    patch_positions = torch.tensor(
        [
            [[0.1, 0.0], [0.3, 0.6]],
            [[0.0, 0.2], [0.4, 0.4]],
        ]
    )
    pos_enc = ImagePositionEncoding(vocab_size=10)
    # Caution: we need to be in eval mode to avoid the random position encoding
    pos_enc.eval()
    pos_encoding = pos_enc(patch_positions)
    torch.testing.assert_close(pos_encoding[0], pos_encoding[1])


def test_image_position_encoding_values_train():
    # The two patches shouldn't have the same encoding because of the random position encoding
    patch_positions = torch.tensor(
        [
            [[0.1, 0.0], [0.3, 0.6]],
            [[0.0, 0.2], [0.4, 0.4]],
        ]
    )
    pos_enc = ImagePositionEncoding(vocab_size=10)
    pos_enc.train()
    pos_encoding = pos_enc(patch_positions)
    assert not torch.allclose(pos_encoding[0], pos_encoding[1])


def test_image_encoder():
    batch_size = 8
    in_channels = 4
    num_res_channels = 32
    out_features = 128
    num_groups = 4
    patch_size = 16

    image_encoder = ImageEncoder(in_channels, num_res_channels, out_features, num_groups, patch_size)
    images = torch.randn(batch_size, in_channels, patch_size, patch_size)
    encoded_images = image_encoder(images)
    assert encoded_images.shape == (
        batch_size,
        out_features,
    ), f"Expected shape ({batch_size}, {out_features}), got {encoded_images.shape}"


def test_embedding_neither_input_ids_nor_patches():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    with pytest.raises(ValueError):
        module()


def test_embedding_both_input_ids_and_patches_but_no_input_types():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 32))
    patches = torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8)
    patch_positions = random_patch_positions((2, 32))
    with pytest.raises(ValueError):
        module(input_ids=input_ids, patches=patches, patch_positions=patch_positions)


def test_embedding_infer_input_types_input_ids():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 32))
    input_types = torch.zeros((2, 32), dtype=torch.long)
    type_infered_output = module(input_ids=input_ids)
    type_specified_output = module(input_ids=input_ids, input_types=input_types)
    torch.testing.assert_close(type_infered_output, type_specified_output)


def test_embedding_infer_input_types_patches():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    # Caution: we need to be in eval mode to avoid the random position encoding
    module.eval()
    patches = torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8)
    patch_positions = random_patch_positions((2, 32))
    input_types = torch.ones((2, 32), dtype=torch.long)
    type_infered_output = module(patches=patches, patch_positions=patch_positions)
    type_specified_output = module(patches=patches, patch_positions=patch_positions, input_types=input_types)
    torch.testing.assert_close(type_infered_output, type_specified_output)


def test_embedding_wrong_input_types():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 32))
    input_types = torch.ones((2, 32), dtype=torch.long)
    input_types[0, 4] = 2  # 2 is not a valid input type
    with pytest.raises(ValueError):
        module(input_ids=input_ids, input_types=input_types)


def test_embedding_missing_patches():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 32))
    input_types = torch.zeros((2, 32), dtype=torch.long)
    input_types[0, 4] = 1  # one patch somewhere in the middle
    with pytest.raises(ValueError):
        module(input_ids=input_ids, input_types=input_types)


def test_embedding_missing_input_ids():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    patches = torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8)
    patch_positions = random_patch_positions((2, 32))
    input_types = torch.ones((2, 32), dtype=torch.long)
    input_types[0, 4] = 0  # one input somewhere in the middle
    with pytest.raises(ValueError):
        module(patches=patches, patch_positions=patch_positions, input_types=input_types)


def test_embedding_patches_without_patch_positions():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    patches = torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8)
    with pytest.raises(ValueError):
        module(patches=patches)


def test_embedding_patch_positions_without_patches():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    patch_positions = random_patch_positions((2, 32))
    with pytest.raises(ValueError):
        module(patch_positions=patch_positions)


@pytest.mark.parametrize("test_mode", ["train", "eval"])
@pytest.mark.parametrize("input_mode", ["input_ids", "patches", "both"])
def test_embedding_attention_mask(test_mode, input_mode):
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    if test_mode == "train":
        module.train()
    else:  # 'eval'
        module.eval()

    attention_mask_1 = torch.randint(0, 2, (2, 32), dtype=torch.bool)
    attention_mask_2 = torch.randint(0, 2, (2, 32), dtype=torch.bool)

    inputs_ids = torch.randint(0, 256, (2, 32)) if input_mode in ["input_ids", "both"] else None
    patches = (
        torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8) if input_mode in ["patches", "both"] else None
    )
    patch_positions = random_patch_positions((2, 32)) if input_mode in ["patches", "both"] else None
    input_types = torch.randint(0, 2, (2, 32), dtype=torch.long) if input_mode in ["both"] else None
    output_1 = module(
        input_ids=inputs_ids,
        patches=patches,
        patch_positions=patch_positions,
        input_types=input_types,
        attention_mask=attention_mask_1,
    )
    output_2 = module(
        input_ids=inputs_ids,
        patches=patches,
        patch_positions=patch_positions,
        input_types=input_types,
        attention_mask=attention_mask_2,
    )

    # In train mode, if the input includes patches, the output shouldn't be the same, even when the attention mask is
    # the same
    # In train mode, if the input doesn't include patches, the output should be the same when the attention mask is
    # the same, but different when the attention mask is different
    # In eval mode, the output should be the same when the attention mask is the same, but different when the
    # attention mask is different
    common_mask = attention_mask_1 & attention_mask_2
    different_mask = attention_mask_1 ^ attention_mask_2
    if test_mode == "train":
        if input_mode in ["patches", "both"]:
            assert not torch.allclose(output_1, output_2)
        else:  # 'input_ids'
            assert torch.allclose(output_1[common_mask], output_2[common_mask], atol=1e-5)
            assert not torch.allclose(output_1[different_mask], output_2[different_mask], atol=1e-5)
    else:  # 'eval'
        assert torch.allclose(output_1[common_mask], output_2[common_mask], atol=1e-5)
        assert not torch.allclose(output_1[different_mask], output_2[different_mask], atol=1e-5)
    # Note that we increase the absolute tolerance to 1e-5 because when having a different attention can change
    # slightly the results, because the batch size changes. See:
    # https://discuss.pytorch.org/t/batch-size-changes-linear-layer-output-values/143706


def test_embedding_local_positions():
    module = Embeddings(embed_dim=128, token_vocab_size=256)
    input_ids = torch.randint(0, 256, (2, 32))
    local_positions = torch.randint(0, 256, (2, 32))
    output_wo_local_positions = module(input_ids=input_ids)
    output_w_local_positions = module(input_ids=input_ids, local_positions=local_positions)
    assert not torch.allclose(output_wo_local_positions, output_w_local_positions)


@pytest.mark.parametrize("embed_dim", [32, 64])
@pytest.mark.parametrize("input_mode", ["input_ids", "patches", "both"])
def test_embeddings_output_shape(embed_dim, input_mode):
    module = Embeddings(embed_dim=embed_dim, token_vocab_size=256)
    inputs_ids = torch.randint(0, 256, (2, 32)) if input_mode in ["input_ids", "both"] else None
    patches = (
        torch.randint(0, 256, (2, 32, 4, 16, 16), dtype=torch.uint8) if input_mode in ["patches", "both"] else None
    )
    patch_positions = random_patch_positions((2, 32)) if input_mode in ["patches", "both"] else None
    input_types = torch.randint(0, 2, (2, 32), dtype=torch.long) if input_mode in ["both"] else None
    output = module(input_ids=inputs_ids, patches=patches, patch_positions=patch_positions, input_types=input_types)
    assert output.shape == (2, 32, embed_dim)
