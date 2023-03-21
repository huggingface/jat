from typing import Dict

import torch
from torch import Tensor, nn
from transformers import AutoImageProcessor, ViTModel


class ImagePositionEncoding(nn.Module):
    """
    A position encoding module for generating row and column positional information for image patches.

    This module calculates the normalized row and column intervals for each patch in an input image, quantizes
    the intervals, and uses them to index learnable row and column position embeddings. During training, a random
    index is uniformly sampled from the quantized interval, whereas during evaluation, the mean of the interval
    is used. The row and column position embeddings are then combined through element-wise addition to get the
    patch position encodings.

    Args:
        vocab_size (int, optional): The size of the position embedding vocabulary. Default is 128.
        patch_size (int, optional): The size of the square patches used to divide the input image. Default is 16.

    Inputs:
        images (torch.Tensor): A tensor of shape (B, C, H, W) containing the input images.
        eval (bool, optional): A flag indicating whether the module is being used for evaluation. Default is False.

    Outputs:
        position_encodings (torch.Tensor): A tensor of shape (B, 1, M, N) containing the patch position encodings
            for each image in the batch.

    Example:
        >>> import torch
        >>> pos_enc = ImagePositionEncoding(vocab_size=128, patch_size=16)
        >>> images = torch.randn(2, 3, 80, 64)
        >>> pos_encoding = pos_enc(images)
        >>> pos_encoding.shape
        torch.Size([2, 1, 5, 4])
    """

    def __init__(self, vocab_size: int = 128, patch_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.row_embedding = nn.Embedding(vocab_size, 1)
        self.column_embedding = nn.Embedding(vocab_size, 1)

    def forward(self, images: Tensor, eval: bool = False) -> Tensor:
        #  First, the relative row and column intervals of the patch are calculated
        # by normalizing the patch’s pixel intervals by the image resolution.
        batch_size, _, height, width = images.size()
        n_rows, n_cols = height // self.patch_size, width // self.patch_size
        norm_row_intervals = torch.arange(n_rows + 1) / n_rows
        norm_col_intervals = torch.arange(n_cols + 1) / n_cols

        # The row and column normalized intervals are then quantized into a vocabulary
        # size (we use 128) and are used to index a row and column table of learnable position encodings.
        quant_row_intervals = (norm_row_intervals * self.vocab_size).round().long()
        quant_col_intervals = (norm_col_intervals * self.vocab_size).round().long()

        # The method in which the quantized row and column intervals are converted into indices depends
        # on whether we are training or evaluating the model: during training a random index is uniformly
        # sampled from the quantized interval, while during evaluation we deterministically take the
        # (rounded) mean of the interval
        sampled_row_idx = torch.zeros((batch_size, n_rows), dtype=torch.long)
        sampled_col_idx = torch.zeros((batch_size, n_cols), dtype=torch.long)
        if eval:
            sampled_row_idx[:] = ((quant_row_intervals[:-1] + quant_row_intervals[1:]) / 2).round()
            sampled_col_idx[:] = ((quant_col_intervals[:-1] + quant_col_intervals[1:]) / 2).round()
        else:
            for row_idx in range(n_rows):
                sampled_row_idx[:, row_idx] = torch.randint(
                    quant_row_intervals[row_idx], quant_row_intervals[row_idx + 1], size=(batch_size,)
                )
            for col_idx in range(n_cols):
                sampled_col_idx[:, col_idx] = torch.randint(
                    quant_col_intervals[col_idx], quant_col_intervals[col_idx + 1], size=(batch_size,)
                )

        # The row and column indices are then used to look up the position encodings in the row and column tables.
        row_pos_encodings = self.row_embedding(sampled_row_idx)
        col_pos_encodings = self.column_embedding(sampled_col_idx)

        # Element-wise addition of the row and column position encodings to get the patch position encodings.
        row_pos_encodings = row_pos_encodings.view(batch_size, 1, n_rows, 1)
        col_pos_encodings = col_pos_encodings.view(batch_size, 1, 1, n_cols)
        return row_pos_encodings + col_pos_encodings


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.groupnorm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.groupnorm2 = nn.GroupNorm(groups, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.groupnorm1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.groupnorm2(out)

        identity = self.shortcut(identity)
        out += identity
        out = F.gelu(out)

        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride, groups=32):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        self.layers = nn.Sequential(
            *[
                BasicBlock(in_channels if i == 0 else out_channels, out_channels, stride=s, groups=groups)
                for i, s in enumerate(strides)
            ]
        )

    def forward(self, x):
        return self.layers(x)


class Embeddings(nn.Module):
    """
    Embedding layer.

    Args:
        embedding_dim (int): The embedding dimension.
        vocab_size (int, optional): The vocabulary size. Defaults to 32_000.
        nb_bins (int, optional): Number of bins. Defaults to 1024.
        max_nb_observation_tokens (int, optional): Maximum number of observation tokens. Defaults to 512.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int = 32_000,
        nb_bins: int = 1024,
        max_nb_observation_tokens: int = 512,
        patch_size: int = 16,
        image_vocab_size: int = 128,
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
        self.action_positional_emb_idx = torch.tensor(max_nb_observation_tokens, dtype=torch.int64)

        self.image_encoder = nn.Conv2d(
            in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.image_pos_enc = ImagePositionEncoding(vocab_size=image_vocab_size, patch_size=patch_size)

        # self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.image_embeddings = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").get_input_embeddings()

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        # Here, d is a dictionary containing the following keys:
        # - "observations[/*]": A tensor of shape (batch_size, L, n_obs_tokens) if observation is not an image.
        #                       A tensor of shape (batch_size, L, num_channels, height, width) if observation is an image.
        # - "actions": A tensor of shape (batch_size, L, n_action_tokens)
        # Where L is the number of interactions in the batch. Note that this number can vary from batch to batch because
        # if the prompt is too short, L will be smaller than the maximum possible number interactions in the batch.

        # First, handle tokens
        observation_keys = [key for key in batch.keys() if key.startswith("observations")]
        tokenized_observation_keys = [key for key in observation_keys if batch[key][0].dim() == 2]
        image_observation_keys = [key for key in observation_keys if batch[key][0].dim() == 4]
        assert len(tokenized_observation_keys) + len(image_observation_keys) == len(observation_keys)

        # Concat all tokenized observations
        tokens = torch.cat([batch[key] for key in tokenized_observation_keys], dim=2)
        embeddings = self.embeddings(tokens)
        observation_pos_emb_idxs = torch.arange(embeddings.shape[1]).unsqueeze(0).repeat(embeddings.shape[0], 1)
        observation_pos_embeddings = self.positional_emb(observation_pos_emb_idxs)
        embeddings += observation_pos_embeddings

        # Now, handle images
        # First, normalize the images to be in [-1, 1]
        images = {key: batch[key].float() * 2.0 / 255.0 - 1.0 for key in image_observation_keys}
        embed_images = {
            key: self.image_encoder(images[key]) + self.image_pos_enc(images[key]) for key in image_observation_keys
        }
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

    dataset = load_gia_dataset("babyai-go-to").shuffle()
    dataloader = DataLoader(dataset, batch_size=16)
    embeddings = Embeddings(512)
    for batch in tqdm(dataloader):
        print(embeddings(batch))
