from torch.utils.data import DataLoader
from tqdm import tqdm

from gia.datasets import load_batched_dataset
from gia.model.embedding import Embeddings

dataset = load_batched_dataset("babyai-go-to")
dataloader = DataLoader(dataset)
embeddings = Embeddings()
for batch in tqdm(dataloader):
    emb = embeddings(batch)
    for key, value in emb.items():
        tqdm.write(f"{key}: {value.shape} {value.dtype}")
    tqdm.write("---")