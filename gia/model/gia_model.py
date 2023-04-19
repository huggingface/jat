from typing import Dict, List

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM

from gia.config import ModelArguments
from gia.model.embedding import Embeddings


class GiaModel(nn.Module):
    def __init__(self, args: ModelArguments) -> None:
        super().__init__()
        if args.use_pretrained:
            raise NotImplementedError("Pretrained models are not supported yet.")
            # self.model = AutoModelForCausalLM.from_pretrained(args.model_name)
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            config.vocab_size = args.text_vocab_size + args.nb_bins
            if args.use_separator:
                config.vocab_size += 1
            config.max_position_embeddings = args.seq_len  # this is a workaround for gpt-neo's local self attn
            config.hidden_size = args.embed_dim
            self.model = AutoModelForCausalLM.from_config(config)

        self.emb = Embeddings(args)

    def forward(self, batch: List[Dict[str, Tensor]], eval=False):
        # hotfix to allow eval in embedding. Try to make it cleaner later
        # add loss_mask to batch
        if eval:
            keys = [key for key in batch.keys() if key.endswith(("observations", "actions"))]
            for key in keys:
                batch[key + "_loss_mask"] = torch.ones_like(batch[key + "_attention_mask"])

        # The batch is a list of dicts, each dict value is a tensor.
        # We need to unsqueeze these tensors to add a batch dimension.
        embeds = []
        for sample in batch:
            for key, value in sample.items():
                sample[key] = value.unsqueeze(0)
            embeds.append(self.emb(sample))

        # embeds is a list of dicts whose keys are "embeddings", "attention_mask", "tokens", "loss_mask"
        # We need to concatenate all the tensors along the batch dimension.
        # FIXME: pretty sure this won't work for multiple size, we need to pad here.
        embeds = {key: torch.cat([embed[key] for embed in embeds], dim=0) for key in embeds[0].keys()}

        # The model requires us to provide position ids, otherwise it will generate them
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
