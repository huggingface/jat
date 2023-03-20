from typing import Dict

import torch
from torch import Tensor, nn
from transformers import AutoImageProcessor, ViTModel


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
        self, embedding_dim: int, vocab_size: int = 32_000, nb_bins: int = 1024, max_nb_observation_tokens: int = 512
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

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_embeddings = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").get_input_embeddings()

    def forward(self, d: Dict[str, Tensor]) -> Tensor:
        # First, handle tokens
        # tokens = d["observation_tokens"]

        # Now, handle images
        images = d["observations/rgb_images"]
        inputs = self.image_processor(images, return_tensors="pt")
        outputs = self.image_embeddings(**inputs)

        observation_size = torch.argmax(tokens[0])  # we assume that the separator token is always the greatest token.
        action_size = tokens.shape[1] - observation_size - 1  # -1 for the separator token

        # Embed tokens
        embeddings = self.embeddings(tokens)

        # Compute and add positional observation embeddings
        observation_pos_emb_idxs = torch.arange(observation_size).unsqueeze(0).repeat(sequence_length, 1)
        observation_pos_embeddings = self.positional_emb(observation_pos_emb_idxs)
        embeddings[:, :observation_size] += observation_pos_embeddings

        # Compute and add positional action embeddings
        action_pos_emb_idxs = self.action_positional_emb_idx.repeat(sequence_length, action_size)
        action_pos_embeddings = self.positional_emb(action_pos_emb_idxs)
        embeddings[:, observation_size + 1 :] += action_pos_embeddings
        return embeddings


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from gia.datasets.gia_dataset import GiaDataset

    dataset = GiaDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    embeddings = Embeddings(512)
    print(next(iter(loader)))

    for batch in loader:
        embeddings(batch)
