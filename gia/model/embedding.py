from typing import Optional

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
        patch_positions (torch.Tensor): A tensor of shape (B, 2, 2) containing the interval of the patch position.
            Each element describes the interval with an array [[x_min, y_min], [x_max, y_max]].
        eval (bool, optional): A flag indicating whether the module is being used for evaluation. Defaults to False.

    Outputs:
        position_encodings (torch.Tensor): A tensor of shape (B, N) containing the patch position encodings
            for each image in the batch.

    Example:
        >>> import torch
        >>> pos_enc = ImagePositionEncoding()
        >>> patch_positions = torch.tensor(
        ...     [
        ...         [[0.0, 0.0], [0.2, 0.3]],
        ...         [[0.1, 0.3], [0.2, 0.4]],
        ...     ]
        ... )
        >>> pos_encoding = pos_enc(patch_positions)
        >>> pos_encoding.shape
        torch.Size([2, 2048])
    """

    def __init__(self, vocab_size: int = 128, embed_dim: int = 2048) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.row_embedding = nn.Embedding(vocab_size, embed_dim)
        self.column_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, patch_positions: Tensor) -> Tensor:
        # The row and column normalized intervals are then quantized into a vocabulary
        # size (we use 128) and are used to index a row and column table of learnable position encodings.
        quant_row_intervals = (patch_positions[..., 0] * self.vocab_size).floor().long()
        quant_col_intervals = (patch_positions[..., 1] * self.vocab_size).floor().long()

        # Edge case (when the high value is 1.0) is handled by setting the high value to vocab_size - 1
        quant_col_intervals[quant_col_intervals == self.vocab_size] = self.vocab_size - 1
        quant_row_intervals[quant_row_intervals == self.vocab_size] = self.vocab_size - 1

        # The method in which the quantized row and column intervals are converted into indices depends
        # on whether we are training or evaluating the model: during training a random index is uniformly
        # sampled from the quantized interval, while during evaluation we deterministically take the
        # (rounded) mean of the interval
        sampled_row_idx = torch.zeros(patch_positions.shape[0], dtype=torch.long, device=patch_positions.device)
        sampled_col_idx = torch.zeros(patch_positions.shape[0], dtype=torch.long, device=patch_positions.device)
        if self.training:
            for idx, (low, high) in enumerate(quant_row_intervals):
                sampled_row_idx[idx] = torch.randint(low, high + 1, (1,))
            for idx, (low, high) in enumerate(quant_col_intervals):
                sampled_col_idx[idx] = torch.randint(low, high + 1, (1,))
        else:
            sampled_row_idx = (quant_row_intervals[..., 0] + quant_row_intervals[..., 1]) // 2
            sampled_col_idx = (quant_col_intervals[..., 0] + quant_col_intervals[..., 1]) // 2

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
        x = self.conv(x)
        x = self.residual_block(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class Embeddings(nn.Module):
    """
    The embedding module for the GIA model.
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
        self.local_pos_embeddings = nn.Embedding(num_local_positions, embed_dim)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        patches: Optional[Tensor] = None,
        patch_positions: Optional[Tensor] = None,
        input_types: Optional[Tensor] = None,
        local_positions: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            patches (`torch.Tensor` of shape `(batch_size, sequence_length, patch_size, patch_size)`, *optional*):
                Tensor containing the image patch data for each image patch in the sequence.
            patch_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Tensor containing the position of each image patch in the sequence.
            input_types (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Tensor indicating the type of each input in the sequence (0 for tokens and 1 for image patches).
            local_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of local positions of each token in the sequence.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[True, False]`:

                - True for tokens that are **not masked**,
                - False for tokens that are **masked**.

        Returns:
            torch.Tensor: The embedding tensor for the input sequence.

        Raises:
            ValueError: If one of the following conditions is met:
                - both input_ids and patches are None
                - input_ids and patches are provided but input_types is None
                - the input_types tensor contains values other than 0 or 1
                - patches is provided but patch_positions is None (and vice-versa)
        """

        # Either input_ids or patches must be provided
        if input_ids is None and patches is None:
            raise ValueError("Either input_ids or patches must be provided.")

        # Get the batch_size and seq_len
        batch_size, seq_len = input_ids.shape if input_ids is not None else patches.shape[:2]

        # Get the device
        device = self.embeddings.weight.device

        # Get the attention mask, if not provided, create one
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.bool()

        # Check input_types
        if input_types is None:
            # Is this case, we need to infer it
            if input_ids is not None and patches is None:
                input_types = torch.zeros((batch_size, seq_len), dtype=torch.long)
            elif input_ids is None and patches is not None:
                input_types = torch.ones((batch_size, seq_len), dtype=torch.long)
            else:
                raise ValueError("When both input_ids and patches are provided, input_types must be provided too.")
        else:
            # Check the provided input_ids
            # Check that all values are etiher 0 or 1
            if not torch.all((input_types[attention_mask] == 0) | (input_types[attention_mask] == 1)):
                raise ValueError("input_types must be either 0 or 1.")
            # Check that if some values are 0, then input_ids is not None
            if torch.any(input_types == 0) and input_ids is None:
                raise ValueError("input_types is 0 but input_ids is None.")
            # Check that if some values are 1, then patches is not None
            if torch.any(input_types == 1) and patches is None:
                raise ValueError("input_types is 1 but patches is None.")

        # Check that patches and patch_positions are provided together
        if (patches is None) != (patch_positions is None):
            raise ValueError("patches and patch_positions must be provided together.")


        # Initialize the embeddings with zeros
        embed = torch.zeros(batch_size, seq_len, self.embed_dim, dtype=torch.float32, device=device)

        # Set the embeddings for the tokens
        if input_ids is not None:
            mask = torch.logical_and(input_types == 0, attention_mask)
            embed[mask] = self.embeddings(input_ids[mask])

        # Set the embeddings for the image patches
        if patches is not None:
            mask = torch.logical_and(input_types == 1, attention_mask)
            normalized_images = patches[mask].float() * 2.0 / 255.0 - 1.0
            embed[mask] = self.image_encoder(normalized_images) + self.image_pos_enc(patch_positions[mask])

        # Add the local position embeddings
        if local_positions is not None:
            mask = torch.logical_and(local_positions != -1, attention_mask)
            embed[mask] = embed[mask] + self.local_pos_embeddings(local_positions[mask])

        return embed
