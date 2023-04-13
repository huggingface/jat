from gia.config import Arguments
from gia.datasets import get_dataloader
from gia.model.embedding import Embeddings

args = Arguments(task_names=["babyai-go-to"], embed_dim=128)
dataloader = get_dataloader(args)
model = Embeddings(args)
batch = next(iter(dataloader))
output = model(batch)
print(output.keys())  # dict_keys(['embeddings', 'loss_mask', 'attention_mask', 'tokens'])
print(output["embeddings"].shape)  # torch.Size([8, 1014, 128])
