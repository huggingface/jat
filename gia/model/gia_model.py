from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

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
        # self.wpe = nn.Embedding(model_config.max_position_embeddings, model_config.embed_dim)

    def forward(self, batch):
        # TODO: spliting based on image patches etc
        # for more flexibility we may want to use our own position ids and zero & freeze the base models position embs
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
        out.loss = self.loss(out.logits, batch["tokens"], batch["loss_mask"])
        return out

    def loss(self, logits, tokens, masks):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        truncated_logits = logits[..., :-1, :].contiguous()
        shifted_tokens = tokens[..., 1:].contiguous()
        truncated_masks = masks[..., 1:].contiguous()

        loss = (
            loss_fn(
                truncated_logits.view(-1, truncated_logits.size(-1)),
                shifted_tokens.view(-1),
            )
            * truncated_masks.view(-1).float()
        )

        loss = loss.sum() / masks.float().sum()

        return loss


if __name__ == "__main__":
    args = Arguments()
    args.tasks = ["mujoco"]
    args.use_cache = True

    dataset = GiaDataset(args)
    dataloader = DataLoader(dataset, batch_size=2)
    model = GiaModel(args)

    batch = next(iter(dataloader))
    out = model(batch)
    print(out)
