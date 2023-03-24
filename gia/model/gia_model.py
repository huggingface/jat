from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from gia.config import Arguments
from gia.model.embedding import Embeddings


class GiaModel(nn.Module):
    def __init__(self, args: Arguments):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name)
        config.vocab_size = args.vocab_size
        config.max_position_embeddings = args.seq_length  # this is a workaround for gpt-neo's local self attn
        self.model = AutoModelForCausalLM.from_config(config)
        config.embed_dim = self.model.base_model.embed_dim
        self.emb = Embeddings(config.embed_dim)

    def forward(self, batch):
        embeds = self.emb(batch)
        # the model requires us to provide position ids, otherwise it will generate them
        # qgallouedec: I've removed position_ids=batch["local_position_ids"]. Is this a problem?
        out = self.model(inputs_embeds=embeds["embeddings"], attention_mask=embeds["attention_mask"])
        out.loss = self.loss(out.logits, embeds["tokens"], embeds["loss_mask"])
        return out

    def loss(self, logits, tokens, masks):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        truncated_logits = logits[..., :-1, :].contiguous()
        shifted_tokens = tokens[..., 1:].contiguous()
        truncated_masks = masks[..., 1:].contiguous()

        loss = loss_fn(
            truncated_logits.view(-1, truncated_logits.size(-1)),
            shifted_tokens.view(-1),
        )
        loss = loss * truncated_masks.view(-1).float()
        loss = loss.sum() / masks.float().sum()

        return loss


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from gia.datasets.gia_dataset import load_gia_dataset

    dataset = load_gia_dataset("mujoco-ant", load_from_cache_file=False)
    dataloader = DataLoader(dataset)
    model = GiaModel(Arguments())
    for batch in tqdm(dataloader):
        out = model(batch)
        tqdm.write(str(out))
