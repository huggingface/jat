import copy

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from gia.config import ModelArguments
from gia.model.embedding import Embeddings


class GiaModel(nn.Module):
    def __init__(self, args: ModelArguments):
        super().__init__()
        vocab_size = args.text_vocab_size + args.nb_bins
        if args.use_separator:
            vocab_size += 1
        if args.use_pretrained:
            config = AutoConfig.from_pretrained(args.model_name)
            config.vocab_size = vocab_size
            config.max_position_embeddings = args.seq_len  # this is a workaround for gpt-neo's local self attn
            self.model = AutoModelForCausalLM.from_config(config)
        else:
            raise NotImplementedError("Training from scratch is not implemented yet.")

        if args.embed_dim == -1:  # automatically set the embedding dimension
            args = copy.copy(args)
            args.embed_dim = self.model.base_model.embed_dim

        if self.model.base_model.embed_dim != args.embed_dim:
            raise ValueError(
                f"The embedding dimension ({args.embed_dim}) must be the same as "
                f"the model ({self.model.base_model.embed_dim}). Use either --embed_dim -1 or "
                f"--embed_dim {self.model.base_model.embed_dim}."
            )

        self.emb = Embeddings(args)

    def forward(self, batch, eval=False):
        # hotfix to allow eval in embedding. Try to make it cleaner later
        # add loss_mask to batch
        if eval:
            keys = [key for key in batch.keys() if key.endswith(("observations", "actions"))]
            for key in keys:
                batch[key + "_loss_mask"] = torch.ones_like(batch[key + "_attention_mask"])

        embeds = self.emb(batch)
        # the model requires us to provide position ids, otherwise it will generate them
        # qgallouedec: I've removed position_ids=batch["local_position_ids"]. Is this a problem?
        out = self.model(inputs_embeds=embeds["embeddings"], attention_mask=embeds["attention_mask"])
        if not eval:
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
