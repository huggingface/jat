from torch import nn

from transformers import AutoModelForCausalLM, AutoConfig
from torch.utils.data.dataloader import DataLoader
from gia.config import Arguments
from gia.datasets import GiaDataset

# class PatchedFeatureExtractor(nn.Module):
#     def __init__(self, args):
#         # create the model etc
#         pass

#     def forward(self, x):
#         # TODO: select image patches from batch, with batch[image_patches] ?
#         # extracted embeddings add to batch
#         return x


class CustomEmbeddingModule(nn.Module):
    def __init__(self, args: Arguments, model_config):
        super().__init__()
        self.wte = nn.Embedding(model_config.vocab_size, model_config.embed_dim)
        # self.wpe = nn.Embedding(model_config.max_position_embeddings, model_config.embed_dim) # see comment in forward

    def forward(self, batch):
        # TODO: spliting based on image patches etc
        # for more flexibility we may want to use our own position ids and zero and freeze the base models position embeddings
        x = self.wte(batch["tokens"])  # + self.wpe(batch["local_position_ids"])
        return x


class GiaModel(nn.Module):
    def __init__(self, args: Arguments):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name)
        config.vocab_size = args.vocab_size
        config.max_position_embeddings = args.seq_length  # this is a workaround for gpt-neo's local self attn
        self.model = AutoModelForCausalLM.from_config(config)
        config.embed_dim = self.model.base_model.embed_dim

        self.emb = CustomEmbeddingModule(args, config)

    def forward(self, batch):
        embeds = self.emb(batch)
        # the model requires us to provide position ids, otherwise it will generate them
        out = self.model(inputs_embeds=embeds, position_ids=batch["local_position_ids"])

        return out


if __name__ == "__main__":
    from tqdm import tqdm

    args = Arguments()
    args.tasks = ["mujoco"]

    dataset = GiaDataset(args)
    dataloader = DataLoader(dataset)
    model = GiaModel(args)

    for i, batch in enumerate(tqdm(dataloader)):
        model(batch)
