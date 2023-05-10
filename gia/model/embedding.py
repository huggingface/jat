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
        embed_dim (int, optional): The embedding dimension. Defaults to 512.

    Inputs:
        patch_pos (torch.Tensor): A tensor of shape (B, 2, 2) containing the interval of the patch positions.
            Each element describes the interval with an array [[x_min, y_min], [x_max, y_max]].
        eval (bool, optional): A flag indicating whether the module is being used for evaluation. Defaults to False.

    Outputs:
        position_encodings (torch.Tensor): A tensor of shape (B, N) containing the patch position encodings
            for each image in the batch.

    Example:
        >>> import torch
        >>> pos_enc = ImagePositionEncoding()
        >>> positions = torch.tensor(
        ...     [
        ...         [[0.0, 0.0], [0.2, 0.3]],
        ...         [[0.1, 0.3], [0.2, 0.4]],
        ...     ]
        ... )
        >>> pos_encoding = pos_enc(positions)
        >>> pos_encoding.shape
        torch.Size([2, 2048])
    """

    def __init__(self, vocab_size: int = 128, embed_dim: int = 2048) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.row_embedding = nn.Embedding(vocab_size, embed_dim)
        self.column_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, patch_pos: Tensor, eval: bool = False) -> Tensor:
        # The row and column normalized intervals are then quantized into a vocabulary
        # size (we use 128) and are used to index a row and column table of learnable position encodings.
        quant_row_intervals = (patch_pos[..., 0] * self.vocab_size).floor().long()
        quant_col_intervals = (patch_pos[..., 1] * self.vocab_size).floor().long()

        # Edge case (when the high value is 1.0) is handled by setting the high value to vocab_size - 1
        quant_col_intervals[quant_col_intervals == self.vocab_size] = self.vocab_size - 1
        quant_row_intervals[quant_row_intervals == self.vocab_size] = self.vocab_size - 1

        # The method in which the quantized row and column intervals are converted into indices depends
        # on whether we are training or evaluating the model: during training a random index is uniformly
        # sampled from the quantized interval, while during evaluation we deterministically take the
        # (rounded) mean of the interval
        sampled_row_idx = torch.zeros(patch_pos.shape[0], dtype=torch.long, device=patch_pos.device)
        sampled_col_idx = torch.zeros(patch_pos.shape[0], dtype=torch.long, device=patch_pos.device)
        if eval:
            sampled_row_idx = (quant_row_intervals[..., 0] + quant_row_intervals[..., 1]) // 2
            sampled_col_idx = (quant_col_intervals[..., 0] + quant_col_intervals[..., 1]) // 2
        else:
            for idx, (low, high) in enumerate(quant_row_intervals):
                sampled_row_idx[idx] = torch.randint(low, high + 1, (1,))
            for idx, (low, high) in enumerate(quant_col_intervals):
                sampled_col_idx[idx] = torch.randint(low, high + 1, (1,))

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

    """
    An image encoder module for extracting image features from a batch of images.

    Args:
        in_channels (int): The number of channels in the input images. If the input images have less channels, they
            are padded with zeros.
        num_res_channels (int): The number of channels in the residual blocks.
        out_features (int): The number of features in the output image features.
        num_groups (int): The number of groups for the GroupNorm layers.
        patch_size (int): The size of the patches to be extracted from the input images.

    Structure:

    ```
          Input image, shape (N, patch_size, patch_size)
               |
          Pad with zeros to shape (in_channels, patch_size, patch_size)
               |
          Conv2d(in_channels, num_res_channels, kernel_size=1)
        _____  |
        |      |
        | GroupNorm(num_groups, num_res_channels)
        |      |
        | Conv2d(num_res_channels, num_res_channels, kernel_size=3, padding=1)
        |      |
        | GroupNorm(num_groups, num_res_channels)
        |      |
        | Conv2d(num_res_channels, num_res_channels, kernel_size=3, padding=1)
        |______|
               |
            Flatten()
               |
            Linear(num_res_channels * patch_size * patch_size, out_features)
               |
    ```
    """

    def __init__(
        self, in_channels: int, num_res_channels: int, out_features: int, num_groups: int, patch_size: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, num_res_channels, kernel_size=1)
        self.residual_block = ResidualBlockV2(num_res_channels, num_groups)
        self.linear = nn.Linear(num_res_channels * patch_size * patch_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # Pad the input images with zeros if they have less channels than the encoder expects
        x = F.pad(x, (0, 0, 0, 0, 0, self.in_channels - x.shape[1]))
        x = self.conv(x)
        x = self.residual_block(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class LocalPositionEncodings(nn.Module):
    """
    A module for computing local position encodings.

    Args:
        vocab_size (int, optional): The size of the vocabulary. Defaults to 128.
        embed_dim (int, optional): The dimension of the embedding. Defaults to 2048.

    Inputs:
        shape (torch.Size): The shape of the input tensor, which should be of the form
            (batch_size, seq_len, num_tokens, embed_dim).
        same (bool, optional): Whether to use the same position encodings for all tokens in a sequence.

    Outputs:
        Tensor: The local position encodings of shape (batch_size, seq_len, num_tokens, embed_dim).

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> batch_size, seq_len, num_tokens = 8, 16, 20
        >>> vocab_size, embed_dim = 128, 2048
        >>> pos_enc = LocalPositionEncodings(vocab_size, embed_dim)
        >>> shape = torch.Size([batch_size, seq_len, num_tokens, embed_dim])
        >>> pos_emb = pos_enc(shape)
        >>> pos_emb.shape
        torch.Size([8, 16, 20, 2048])
    """

    def __init__(self, vocab_size: int = 128, embed_dim: int = 2048) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, shape: torch.Size, same: bool = False) -> Tensor:
        batch_size, seq_len, num_tokens, embed_dim = shape
        device = self.embedding.weight.device
        assert embed_dim == self.embed_dim
        if same:
            pos_emb_idxs = torch.full((num_tokens,), self.vocab_size - 1, dtype=torch.long, device=device)
        else:
            pos_emb_idxs = torch.arange(num_tokens, device=device)
        pos_emb_idxs = pos_emb_idxs.view(1, 1, num_tokens)
        pos_emb_idxs = pos_emb_idxs.expand(batch_size, seq_len, num_tokens)
        return self.embedding(pos_emb_idxs)


class Embeddings(nn.Module):
    """
    Embeddings layer.

    Attributes:
        embed_dim: The dimension of the output embedding.
        token_vocab_size: The size of the vocabulary for token embeddings.
        num_local_positions: The number of local position encodings for the sequence.
        patch_size: The size of image patches.
        image_vocab_size: The number of unique image patches that can be encoded.
        num_res_channels: The number of residual channels in the image encoder.
        num_groups: The number of groups to use in the grouped convolution in the image encoder.

    Examples:
        >>> PATCH_PAD = torch.zeros((4, 16, 16), dtype=torch.float32).tolist() # tolist inefficient but easier to read
        >>> PATCH = torch.rand(4, 16, 16, dtype=torch.float32).tolist()
        >>> POS_PAD = [[0.0, 0.0], [0.0, 0.0]]
        >>> POS = [[0.2, 0.2], [0.5, 0.6]]
        >>> embed = Embeddings(embed_dim=4, token_vocab_size=10)
        >>> input_ids = torch.tensor([[1, 0, 2], [4, 0, 0]])
        >>> patches = torch.tensor([[PATCH_PAD, PATCH, PATCH_PAD], [PATCH_PAD, PATCH_PAD, PATCH_PAD]])
        >>> positions = torch.tensor([[POS_PAD, POS, POS_PAD], [POS_PAD, POS_PAD, POS_PAD]])
        >>> input_type = torch.tensor([[0, 1, 0], [0, 0, 0]])
        >>> attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
        >>> print(embed(input_ids, patches, positions, input_type, attention_mask))
        tensor([[[ 0.6603, -1.4121,  1.8346,  0.4562],
                 [ 0.1788,  0.8613,  0.6966,  0.1318],
                 [ 0.2750,  0.1993,  0.1091, -0.8430]],

                [[ 0.6289,  1.0134, -0.7994, -0.1257],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000]]], grad_fn=<IndexPutBackward0>)
    """

    def __init__(
        self,
        embed_dim: int,
        token_vocab_size: int,
        num_local_positions: int = 512,
        patch_size: int = 16,
        image_vocab_size: int = 128,
        num_res_channels: int = 256,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Embedding layer for the tokens
        self.embeddings = nn.Embedding(token_vocab_size, embed_dim)

        # Encoder for the image patches
        self.image_encoder = ImageEncoder(4, num_res_channels, embed_dim, num_groups, patch_size)

        # Learnable local position encodings for the image patches
        self.image_pos_enc = ImagePositionEncoding(image_vocab_size, embed_dim)

        # Learnable local position encodings in the sequence
        # The total number of tokens is the number of observation tokens + 1 for the unique action token
        self.local_pos_embeddings = LocalPositionEncodings(num_local_positions, embed_dim)

    def forward(self, input_ids, patches, positions, input_type, attention_mask) -> Tensor:
        device = self.embeddings.weight.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        embed = torch.zeros(batch_size, seq_len, self.embed_dim, dtype=torch.float32, device=device)
        mask = torch.logical_and(input_type == 0, attention_mask)
        embed[mask] = self.embeddings(input_ids[mask])

        mask = torch.logical_and(input_type == 1, attention_mask)
        normalized_images = patches[mask].float() * 2.0 / 255.0 - 1.0
        embed[mask] = self.image_encoder(normalized_images) + self.image_pos_enc(positions[mask])

        return embed
