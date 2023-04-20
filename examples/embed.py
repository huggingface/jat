from torch.utils.data import DataLoader

from gia.config import Arguments
from gia.datasets import collate_fn, load_mixed_dataset
from gia.model.embedding import Embeddings

args = Arguments(task_names=["babyai-go-to"], embed_dim=128)
dataset = load_mixed_dataset(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
embeddings = Embeddings(args)
batch = next(iter(dataloader))
embeds = [embeddings(sample) for sample in batch]
print(len(embeds))  # 8 (= batch_size)
print(embeds[0].keys())  # dict_keys(['embeddings', 'loss_mask', 'attention_mask', 'tokens'])
