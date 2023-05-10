from typing import List

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from gia.config import ModelArguments
from gia.model.embedding import Embeddings


def pad_and_cat(tensor_list: List[Tensor], max_len: int) -> Tensor:
    """
    Pad right with zeros, in the 2nd dimension and concatenate.

    Args:
        tensor_list (List[Tensor]): List of tensors to pad. Shapes must be (N, X, ...) where X can vary.
        max_len (int): The output tensor will have the shape (N, max_len, ...).

    Returns:
        Tensor: The padded and concatenated tensor.

    Example:
        >>> x = [torch.rand(1, 10, 3), torch.rand(1, 5, 3), torch.rand(1, 15, 3)]
        >>> pad_and_cat(x, 20).shape
        torch.Size([3, 20, 3])
    """
    output_shape = [len(tensor_list), max_len, *tensor_list[0].shape[2:]]
    output = torch.zeros(output_shape, dtype=tensor_list[0].dtype, device=tensor_list[0].device)
    for i, t in enumerate(tensor_list):
        output[i, : t.shape[1]] = t
    return output


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

    def forward(self, input_ids, patches, positions, input_type, loss_mask, attention_mask) -> CausalLMOutputWithPast:
        embeds = self.emb(input_ids, patches, positions, input_type, loss_mask, attention_mask)
        labels = embeds["tokens"]
        # All labels set to -100 are ignored (masked), the loss is only computed for labels in
        # [0, ..., config.vocab_size]
        labels[loss_mask == 0] = -100
        return self.model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
