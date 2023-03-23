from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ImagePositionEncoding(nn.Module):
    """
    A position encoding module for generating row and column positional information for image patches.

    This module calculates the normalized row and column intervals for each patch in an input image, quantizes
    the intervals, and uses them to index learnable row and column position embeddings. During training, a random
    index is uniformly sampled from the quantized interval, whereas during evaluation, the mean of the interval
    is used. The row and column position embeddings are then combined through element-wise addition to get the
    patch position encodings.

    Args:
        vocab_size (int, optional): The size of the position embedding vocabulary. Defaults to 128.
        embedding_dim (int, optional): The embedding dimension. Defaults to 512.

    Inputs:
        positions (torch.Tensor): A tensor of shape (B, 2) containing the positions of the patches.
        eval (bool, optional): A flag indicating whether the module is being used for evaluation. Defaults to False.

    Outputs:
        position_encodings (torch.Tensor): A tensor of shape (B, N) containing the patch position encodings
            for each image in the batch.

    Example:
        >>> import torch
        >>> pos_enc = ImagePositionEncoding(vocab_size=128)
        >>> images = torch.rand(10, 2)
        >>> pos_encoding = pos_enc(images)
        >>> pos_encoding.shape
        torch.Size([10, 128])
    """

    def __init__(self, vocab_size: int = 128, embedding_dim: int = 2048) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.row_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.column_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, positions: Tensor, eval: bool = False) -> Tensor:
        # The row and column normalized intervals are then quantized into a vocabulary
        # size (we use 128) and are used to index a row and column table of learnable position encodings.
        quant_row_intervals = (positions[..., 0] * self.vocab_size).round().long()
        quant_col_intervals = (positions[..., 1] * self.vocab_size).round().long()

        # The method in which the quantized row and column intervals are converted into indices depends
        # on whether we are training or evaluating the model: during training a random index is uniformly
        # sampled from the quantized interval, while during evaluation we deterministically take the
        # (rounded) mean of the interval
        sampled_row_idx = torch.zeros(positions.shape[0], dtype=torch.long)
        sampled_col_idx = torch.zeros(positions.shape[0], dtype=torch.long)
        if eval:
            sampled_row_idx = (quant_row_intervals[..., 0] + quant_row_intervals[..., 1]) // 2
            sampled_col_idx = (quant_col_intervals[..., 0] + quant_col_intervals[..., 1]) // 2
        else:
            # low == high == 0 happens when timestep is masked, so we need to handle this case
            for idx, (low, high) in enumerate(quant_row_intervals):
                sampled_row_idx[idx] = torch.randint(low, high, (1,)) if low != high else low
            for idx, (low, high) in enumerate(quant_col_intervals):
                sampled_col_idx[idx] = torch.randint(low, high, (1,)) if low != high else low

        # The row and column indices are then used to look up the position encodings in the row and column tables.
        row_pos_encodings = self.row_embedding(sampled_row_idx)
        col_pos_encodings = self.column_embedding(sampled_col_idx)
        return row_pos_encodings + col_pos_encodings


class ResidualBlockV2(nn.Module):
    """
    A residual block with GroupNorm and GELU activations.

    It consists of two convolutional layers with GroupNorm and GELU activations, followed by a residual
    connection.

    Args:
        num_channels (int): The number of channels.
        num_groups (int): The number of groups for the GroupNorm layers.
    """

    def __init__(self, num_channels: int, num_groups: int) -> None:
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups, num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.gn1(x)
        y = F.gelu(y)
        y = self.conv1(y)
        y = self.gn2(y)
        y = F.gelu(y)
        y = self.conv2(y)
        return x + y


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, out_features: int, num_groups: int, patch_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_features, kernel_size=1)
        self.residual_block = ResidualBlockV2(out_features, num_groups)
        self.linear = nn.Linear(out_features * patch_size * patch_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.residual_block(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class MultiDimBatchWrapper(nn.Module):
    """
    A wrapper for a module that allows it to process inputs with an arbitrary number of leading dimensions.

    Args:
        module (nn.Module): The module to wrap.
        n_dims (int, optional): The number of leading dimensions to preserve. Defaults to 2.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> b1, b2, c, h, w = 10, 5, 3, 32, 32
        >>> conv = nn.Conv2d(c, 16, kernel_size=3, padding=1)
        >>> wrapper = MultiDimBatchWrapper(conv, n_dims=2)
        >>> images = torch.randn(b1, b2, c, h, w)
        >>> conv(images).shape
        torch.Size([10, 5, 16, 32, 32])
    """

    def __init__(self, module: nn.Module, n_dims: int = 2) -> None:
        super().__init__()
        self.module = module
        self.n_dims = n_dims

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[: self.n_dims]
        x = x.view(-1, *x.shape[self.n_dims :])
        x = self.module(x)
        x = x.view(*batch_shape, *x.shape[1:])
        return x


class Embeddings(nn.Module):
    """
    Embedding layer.

    Args:
        embedding_dim (int, optional): The embedding dimension. Defaults to 2048.
        vocab_size (int, optional): The vocabulary size. Defaults to 32_000.
        nb_bins (int, optional): Number of bins for the discretization of continuous values. Defaults to 1024.
        max_nb_observation_tokens (int, optional): Maximum number of observation tokens. Defaults to 512.
        patch_size (int, optional): The size of the square patch to be extracted from the image. Defaults to 16.
        image_vocab_size (int, optional): The size of the position embedding vocabulary for the image
            patches. Defaults to 128.
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        vocab_size: int = 32_000,
        nb_bins: int = 1024,
        max_nb_observation_tokens: int = 512,
        patch_size: int = 16,
        image_vocab_size: int = 128,
    ) -> None:
        super().__init__()
        self.separator_token = nb_bins + vocab_size
        # The total number of tokens is the number of observation tokens + 1 for the separator token.
        self.embeddings = nn.Embedding(num_embeddings=vocab_size + nb_bins + 1, embedding_dim=embedding_dim)
        # Learnable local position encodings
        # The total number of tokens is the number of observation tokens + 1 for the unique action token
        # TODO: It's not clear whether we should add postional embeddings for the separator token.
        # For now, we don't add it.
        self.positional_emb = nn.Embedding(num_embeddings=max_nb_observation_tokens + 1, embedding_dim=embedding_dim)
        self.action_positional_emb_idx = torch.tensor(max_nb_observation_tokens, dtype=torch.int64)

        self.image_encoder = MultiDimBatchWrapper(ImageEncoder(3, embedding_dim, num_groups=32, patch_size=16), 3)
        self.image_pos_enc = MultiDimBatchWrapper(ImagePositionEncoding(vocab_size=image_vocab_size), 3)

        # self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.image_embeddings = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").get_input_embeddings()

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        # Here, d is a dictionary containing the following keys:
        # - "observations[/*]": A tensor of shape (batch_size, L, n_obs_tokens) if observation is not an image.
        #                       A tensor of shape (batch_size, L, num_patches, num_channels, height, width) if observation is an image.
        # - "actions": A tensor of shape (batch_size, L, n_action_tokens)
        # Where L is the number of interactions in the batch. Note that this number can vary from batch to batch because
        # if the prompt is too short, L will be smaller than the maximum possible number interactions in the batch.

        # First, handle tokens
        observation_keys = [key for key in batch.keys() if key.startswith("observations") and not key.endswith("mask")]
        tokenized_observation_keys = [key for key in observation_keys if batch[key].dim() == 3]
        image_observation_keys = [key for key in observation_keys if batch[key].dim() == 6]
        assert len(tokenized_observation_keys) + len(image_observation_keys) == len(observation_keys)

        # Concat all tokenized observations
        tokens = torch.cat([batch[key] for key in tokenized_observation_keys], dim=2)
        embeddings = self.embeddings(tokens)  # shape (batch_size, L, n_obs_tokens, embedding_dim)
        obs_pos_emb_idxs = torch.arange(embeddings.shape[1])
        # Expand the position embedding indices to match the batch size, then the number of observation tokens
        obs_pos_emb_idxs = obs_pos_emb_idxs.unsqueeze(0).repeat(embeddings.shape[0], 1)
        obs_pos_emb_idxs = obs_pos_emb_idxs.unsqueeze(2).repeat(1, 1, embeddings.shape[2])
        observation_pos_embeddings = self.positional_emb(obs_pos_emb_idxs)
        embeddings += observation_pos_embeddings

        # Now, handle images
        # First, normalize the images to be in [-1, 1]
        dict_images = {key: batch[key].float() * 2.0 / 255.0 - 1.0 for key in image_observation_keys}
        positions = {key: batch[f"_positions/{key}"] for key in image_observation_keys}
        embed_images = {key: self.image_encoder(images) for key, images in dict_images.items()}
        patch_pos_embeddings = {key: self.image_pos_enc(positions[key]) for key in image_observation_keys}
        # Reshape to the right size: from (batch_size, embedding_dim, n_rows, n_cols) to (batch_size, n_rows * n_cols, embedding_dim)
        embed_images = {
            key: embed_images[key]
            .permute(0, 2, 3, 1)
            .reshape(embed_images[key].shape[0], -1, embed_images[key].shape[1])
            for key in image_observation_keys
        }

        # Concat all observations
        embeddings = torch.cat([embeddings] + [embed_images[key] for key in image_observation_keys], dim=1)

        # Compute and add positional action embeddings
        action_pos_emb_idxs = self.action_positional_emb_idx.repeat(sequence_length, action_size)
        action_pos_embeddings = self.positional_emb(action_pos_emb_idxs)
        embeddings[:, observation_size + 1 :] += action_pos_embeddings
        return embeddings


if __name__ == "__main__":
    # TODO: This still does not work because the concatenation of observations and actions is not properly
    # implemented in the dataset.
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from gia.datasets.gia_dataset import load_gia_dataset

    dataset = load_gia_dataset("babyai-go-to", load_from_cache_file=False)
    dataloader = DataLoader(dataset, batch_size=1)
    embeddings = Embeddings(256)
    for batch in tqdm(dataloader):
        print(embeddings(batch))
