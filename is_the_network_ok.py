from torch import nn


class MyEncoder(nn.Module):
    def __init__(self, input_size, embedding_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=148, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=input_size * embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(-1)
        x = self.linear(x)
        return x
    
class MyDecoder(nn.Module):
    def __init__(self, output_size, embedding_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # Assuming that the encoder's linear layer output size is `output_size`
        self.linear = nn.Linear(in_features=output_size, out_features=embedding_dim*30)
        # Additional layer to reconstruct the original input
        self.output_layer = nn.Linear(in_features=embedding_dim*30, out_features=output_size)
        
    def forward(self, x):
        x = self.linear(x)
        # Reshape the output to [batch_size, seq_length, embedding_dim]
        x = x.view(-1, 148, self.embedding_dim)
        # You might apply some operation to get back to the original input
        # For example, if the original input was one-hot encoded, you might apply a softmax operation
        x = self.output_layer(x)
        return x


from datasets import load_dataset
dataset = load_dataset("gia-project/gia-dataset-parquet", "babyai-go-to", split="train")

def func(examples):
    flat = [obs for ep in examples["discrete_observations"] for obs in ep]
    return {"discrete_observations": flat}

dataset = dataset.map(func, batched=True, remove_columns=["text_observations", "rewards", "discrete_actions"])

print(len(dataset))