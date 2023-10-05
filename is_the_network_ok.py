import torch
import torch.optim as optim
from datasets import load_dataset, concatenate_datasets
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class MyEncoder(nn.Module):
    def __init__(self, input_size, num_logits, embedding_dim, hidden_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_logits, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=input_size * embedding_dim, out_features=hidden_size)

    def forward(self, x):
        x = self.embedding(x)  # B x S x E
        x = torch.flatten(x, start_dim=-2)  # B x (S*E)
        x = self.linear(x)  # B x H
        return x


class MyDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim, input_size, num_logits) -> None:
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(in_features=hidden_size, out_features=embedding_dim * input_size)
        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=num_logits)

    def forward(self, x):
        x = self.linear(x)  # B x (S*E)
        x = torch.unflatten(x, dim=-1, sizes=(self.input_size, self.embedding_dim))  # B x S x E
        x = self.output_layer(x)  # B x S x num_logits
        return x


tasks = ["babyai-go-to", "babyai-pickup", "babyai-synth", "babyai-boss-level", "babyai-boss-level-no-unlock"]

dataset = concatenate_datasets(
    [load_dataset("gia-project/gia-dataset-parquet", task, split="train").select(range(2000)) for task in tasks]
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# First, convert dataset of episodes into dataset of transitions, then, tokenize the text observations
# and concatenate them with the discrete observations
def func(examples):
    discrete = [obs for ep in examples["discrete_observations"] for obs in ep]
    text = [obs for ep in examples["text_observations"] for obs in ep]
    tokens = tokenizer(text, padding="max_length", max_length=43)["input_ids"]
    # Print the maximum length of the text observations
    print(max(len(obs) for obs in text))
    return {"discrete_observations": [a + b for a, b in zip(discrete, tokens)]}


dataset = dataset.map(
    func, batched=True, remove_columns=["rewards", "discrete_actions", "text_observations"], num_proc=16
)


num_logits = tokenizer.vocab_size  # number of possible tokens
input_size = len(dataset[0]["discrete_observations"])  # num integers in the input
embedding_dim = 768  # embedding dimension of the discrete values
hidden_size = 768  # bottleneck dimension

encoder = MyEncoder(input_size, num_logits, embedding_dim, hidden_size).to("cuda")
decoder = MyDecoder(hidden_size, embedding_dim, input_size, num_logits).to("cuda")

# Hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32


def collate(batch):
    return torch.tensor([x["discrete_observations"] for x in batch]).to("cuda")


# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for idx, batch in tqdm(enumerate(dataloader)):
        encoder_output = encoder(batch)
        decoder_output = decoder(encoder_output)
        loss = criterion(decoder_output.view(-1, num_logits), batch.view(-1))

        accuracy = (decoder_output.argmax(dim=-1) == batch).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
            )
