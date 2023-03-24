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
        patch_pos (torch.Tensor): A tensor of shape (B, 2, 2) containing the interval of the patch positions.
            Each element describes the interval with an array [[x_min, y_min], [x_max, y_max]].
        eval (bool, optional): A flag indicating whether the module is being used for evaluation. Defaults to False.

    Outputs:
        position_encodings (torch.Tensor): A tensor of shape (B, N) containing the patch position encodings
            for each image in the batch.

    Example:
        >>> import torch
        >>> pos_enc = ImagePositionEncoding()
        >>> images = torch.tensor(
        ...     [
        ...         [[0.0, 0.0], [0.2, 0.3]],
        ...         [[0.1, 0.3], [0.2, 0.4]],
        ...     ]
        ... )
        >>> pos_encoding = pos_enc(images)
        >>> pos_encoding.shape
        torch.Size([2, 2048])
    """

    def __init__(self, vocab_size: int = 128, embedding_dim: int = 2048) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.row_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.column_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, patch_pos: Tensor, eval: bool = False) -> Tensor:
        # The row and column normalized intervals are then quantized into a vocabulary
        # size (we use 128) and are used to index a row and column table of learnable position encodings.
        quant_row_intervals = (patch_pos[..., 0] * self.vocab_size).round().long()
        quant_col_intervals = (patch_pos[..., 1] * self.vocab_size).round().long()

        # The method in which the quantized row and column intervals are converted into indices depends
        # on whether we are training or evaluating the model: during training a random index is uniformly
        # sampled from the quantized interval, while during evaluation we deterministically take the
        # (rounded) mean of the interval
        sampled_row_idx = torch.zeros(patch_pos.shape[0], dtype=torch.long)
        sampled_col_idx = torch.zeros(patch_pos.shape[0], dtype=torch.long)
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

    """
    An image encoder module for extracting image features from a batch of images.

    Args:
        in_channels (int): The number of channels in the input images.
        num_res_channels (int): The number of channels in the residual blocks.
        out_features (int): The number of features in the output image features.
        num_groups (int): The number of groups for the GroupNorm layers.
        patch_size (int): The size of the patches to be extracted from the input images.

    Structure:

    ```
          Input image shape (in_channels, patch_size, patch_size)
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
        self.conv = nn.Conv2d(in_channels, num_res_channels, kernel_size=1)
        self.residual_block = ResidualBlockV2(num_res_channels, num_groups)
        self.linear = nn.Linear(num_res_channels * patch_size * patch_size, out_features)

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
        >>> conv = MultiDimBatchWrapper(conv, n_dims=2)
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


class LocalPositionEncodings(nn.Module):
    """
    A module for computing local position encodings.

    Args:
        vocab_size (int, optional): The size of the vocabulary. Defaults to 128.
        embedding_dim (int, optional): The dimension of the embedding. Defaults to 2048.

    Inputs:
        shape (torch.Size): The shape of the input tensor, which should be of the form
            (batch_size, seq_len, num_tokens, embedding_dim).
        same (bool, optional): Whether to use the same position encodings for all tokens in a sequence.

    Outputs:
        Tensor: The local position encodings of shape (batch_size, seq_len, num_tokens, embedding_dim).

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> batch_size, seq_len, num_tokens = 8, 16, 20
        >>> vocab_size, embedding_dim = 128, 2048
        >>> pos_enc = LocalPositionEncodings(vocab_size, embedding_dim)
        >>> shape = torch.Size([batch_size, seq_len, num_tokens, embedding_dim])
        >>> pos_emb = pos_enc(shape)
        >>> pos_emb.shape
        torch.Size([8, 16, 20, 2048])
    """

    def __init__(self, vocab_size: int = 128, embedding_dim: int = 2048) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, shape: torch.Size, same: bool = False) -> Tensor:
        batch_size, seq_len, num_tokens, embedding_dim = shape
        assert embedding_dim == self.embedding_dim
        if same:
            pos_emb_idxs = torch.full((seq_len,), self.vocab_size - 1, dtype=torch.long)
        else:
            pos_emb_idxs = torch.arange(seq_len)
        pos_emb_idxs = pos_emb_idxs.view(1, seq_len, 1)
        pos_emb_idxs = pos_emb_idxs.expand(batch_size, seq_len, num_tokens)
        return self.embedding(pos_emb_idxs)


class Embeddings(nn.Module):
    """
    Embedding layer.

    Args:
        embedding_dim (int, optional): The embedding dimension. Defaults to 2048.
        text_vocab_size (int, optional): The vocabulary size for text. Defaults to 32_000.
        nb_bins (int, optional): Number of bins for the discretization of continuous values. Defaults to 1024.
        max_nb_observation_tokens (int, optional): Maximum number of observation tokens. Defaults to 512.
        use_separator (bool, optional): Whether to use a separator token. Defaults to True.
        patch_size (int, optional): The size of the square patch to be extracted from the image. Defaults to 16.
        image_vocab_size (int, optional): The size of the position embedding vocabulary for the image
            patches. Defaults to 128.
        num_res_channels (int, optional): The number of residual channels in the image patch encoder. Defaults to 256.
        num_groups (int, optional): The number of groups in the image patch encoder. Defaults to 32.

    Inputs:
        batch (Dict[str, Tensor]): A batch of data. It is expected to contain the observations and the actions.
            The possible keys for observations are "text_observations", "image_observations", "discrete_observations",
            and "continuous_observation".
            The possible keys for actions are "text_actions", "discrete_actions" and "continuous_actions". It is
            expected that the batch contains only one action key.
            For each observation and action key, it is expected that the batch also contains loss mask and the
            attention mask. The corresponding keys are "{key}_loss_mask" and "{key}_attention_mask" respectively.
            For example, for the "text_observations" key, the batch should contain "text_observations_loss_mask" and
            "text_observations_attention_mask".

    Outputs:
        Tensor: The embeddings of shape (batch_size, seq_len*L, embedding_dim) where L is the total number of
            embeddings for each sequence, equal to the sum of the number of tokens (or patches) for each modality.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> batch_size, seq_len, num_tokens = 8, 16, 20
        >>> batch = {
        ...     "observations": torch.randint(0, 32_000, (batch_size, seq_len, num_tokens)),
        ...     "actions": torch.randint(32_000, 32_010, (batch_size, seq_len, num_tokens)),
        ... }
        >>> embed = Embeddings()
        >>> embeddings = embed(batch)
        >>> embeddings.shape
        torch.Size([8, 640, 2048])
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        text_vocab_size: int = 32_000,
        nb_bins: int = 1024,
        max_nb_observation_tokens: int = 512,
        use_separator: bool = True,
        patch_size: int = 16,
        image_vocab_size: int = 128,
        num_res_channels: int = 256,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        # Encoder for tokens
        # The total number of tokens is the number of tokens for text + the max number of bins
        # for the continuous and discrete values + 1 for the separator token
        self.use_separator = use_separator
        if use_separator:
            self.embeddings = nn.Embedding(text_vocab_size + nb_bins + 1, embedding_dim)
            self.separator_token = text_vocab_size + nb_bins
        else:
            self.embeddings = nn.Embedding(text_vocab_size + nb_bins, embedding_dim)

        # Encoder for the image patches
        image_encoder = ImageEncoder(3, num_res_channels, embedding_dim, num_groups, patch_size)
        self.image_encoder = MultiDimBatchWrapper(image_encoder, n_dims=3)

        # Learnable local position encodings for the image patches
        self.image_pos_enc = MultiDimBatchWrapper(ImagePositionEncoding(image_vocab_size, embedding_dim), 3)

        # Learnable local position encodings in the sequence
        # The total number of tokens is the number of observation tokens + 1 for the unique action token
        self.local_pos_embeddings = LocalPositionEncodings(max_nb_observation_tokens + 1, embedding_dim)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        # Here, batch is a dictionary containing the following keys:
        # - "observations[/*]": A tensor of shape (batch_size, L, n_obs_tokens) if observation is not an image.
        #                       A tensor of shape (batch_size, L, num_patches, num_channels, height, width) if
        #                           observation is an image.
        # - "actions": A tensor of shape (batch_size, L, n_action_tokens)
        # Where L is the number of interactions in the batch. Note that this number can vary from batch
        # to batch, because, if the prompt is too short, L is smaller than the maximum possible number
        # interactions in the batch.
        # First, handle observations: get the keys of tokens and images
        possible_tokenized_obs_keys = ["text_observations", "discrete_observations", "continuous_observations"]
        possible_action_keys = ["text_actions", "discrete_actions", "continuous_actions"]
        tokenized_obs_keys = [key for key in batch.keys() if key in possible_tokenized_obs_keys]
        has_image = "image_observations" in batch.keys()
        action_key = [key for key in batch.keys() if key in possible_action_keys][0]

        # Handle images observations: normalize and embed, then add patch position embeddings
        if has_image:
            normalized_images = batch["image_observations"].float() * 2.0 / 255.0 - 1.0
            image_embeddings = self.image_encoder(normalized_images)
            patch_pos_embeddings = self.image_pos_enc(batch["patches_positions"])
            image_embeddings = image_embeddings + patch_pos_embeddings

        # Handle tokens observations: concatenate all tokenized observations and embed
        # FIXME: what happens when there is just image observations?
        tokenized_obs = torch.cat([batch[key] for key in tokenized_obs_keys], dim=2)
        obs_embeddings = self.embeddings(tokenized_obs)  # shape (batch_size, L, n_obs_tokens, embedding_dim)
        obs_loss_mask = torch.cat([batch[f"{key}_loss_mask"] for key in tokenized_obs_keys], dim=2)
        obs_attention_mask = torch.cat([batch[f"{key}_attention_mask"] for key in tokenized_obs_keys], dim=2)

        # Concatenate images embeddings with the other embeddings and add local position embeddings
        if has_image:
            obs_embeddings = torch.cat((obs_embeddings, image_embeddings), dim=2)
            obs_loss_mask = torch.cat((obs_loss_mask, batch["image_observations_loss_mask"]), dim=2)
            obs_attention_mask = torch.cat((obs_attention_mask, batch["image_observations_attention_mask"]), dim=2)
        obs_pos_embeddings = self.local_pos_embeddings(obs_embeddings.shape)
        obs_embeddings += obs_pos_embeddings

        # Create separator token
        batch_size, seq_len, _, embedding_dim = obs_embeddings.shape
        if self.use_separator:
            separator_token = torch.full((batch_size, seq_len, 1), self.separator_token, dtype=torch.long)
            separator_embeddings = self.embeddings(separator_token)
            separator_loss_mask = torch.ones((batch_size, seq_len, 1), dtype=torch.bool)
            separator_attention_mask = torch.ones((batch_size, seq_len, 1), dtype=torch.bool)

        # Handle action: embed and add local position embeddings
        action_embeddings = self.embeddings(batch[action_key])
        action_pos_embeddings = self.local_pos_embeddings(action_embeddings.shape, same=True)
        action_embeddings += action_pos_embeddings
        action_loss_mask = batch[f"{action_key}_loss_mask"]
        action_attention_mask = batch[f"{action_key}_attention_mask"]

        # Concatenate all embeddings
        if self.use_separator:
            embeddings = torch.cat([obs_embeddings, action_embeddings], dim=2)
            loss_mask = torch.cat([obs_loss_mask, action_loss_mask], dim=2)
            attention_mask = torch.cat([obs_attention_mask, action_attention_mask], dim=2)
        else:
            embeddings = torch.cat([obs_embeddings, separator_embeddings, action_embeddings], dim=2)
            loss_mask = torch.cat([obs_loss_mask, separator_loss_mask, action_loss_mask], dim=2)
            attention_mask = torch.cat([obs_attention_mask, separator_attention_mask, action_attention_mask], dim=2)

        # Flatten the embeddings into a single sequence of tokens of shape (batch_size, seq_len*L, embedding_dim)
        # Where L is the total number of tokens for one interaction (observation + separator + action)
        embeddings = embeddings.reshape(batch_size, -1, embedding_dim)
        loss_mask = loss_mask.reshape(batch_size, -1)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return {"embeddings": embeddings, "loss_mask": loss_mask, "attention_mask": attention_mask}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from gia.datasets.gia_dataset import load_gia_dataset

    dataset = load_gia_dataset("babyai-go-to", load_from_cache_file=False)
    dataloader = DataLoader(dataset)
    embeddings = Embeddings()
    for batch in tqdm(dataloader):
        emb = embeddings(batch)
        for key, value in emb.items():
            tqdm.write(f"{key}: {value.shape} {value.dtype}")
        tqdm.write("---")
