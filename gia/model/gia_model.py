from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from gia.config import ModelArguments
from gia.model.embedding import Embeddings


class GiaModel(nn.Module):
    def __init__(self, args: ModelArguments) -> None:
        super().__init__()
        if args.use_separator:
            vocab_size = args.text_vocab_size + args.nb_bins + 1
        else:
            vocab_size = args.text_vocab_size + args.nb_bins
        if args.use_pretrained:
            raise NotImplementedError("Pretrained models are not supported yet.")
            # self.model = AutoModelForCausalLM.from_pretrained(args.model_name)
        else:
            config = AutoConfig.from_pretrained(args.model_name)
            config.vocab_size = vocab_size
            config.max_position_embeddings = args.seq_len  # this is a workaround for gpt-neo's local self attn
            config.hidden_size = args.embed_dim
            self.model = AutoModelForCausalLM.from_config(config)

        self.emb = Embeddings(
            args.embed_dim,
            vocab_size,
            args.max_nb_observation_tokens,
            args.patch_size,
            args.image_vocab_size,
            args.num_res_channels,
            args.num_groups,
        )

    def forward(
        self, input_ids, local_positions, patches, patch_positions, input_types, loss_mask, attention_mask
    ) -> CausalLMOutputWithPast:
        embeds = self.emb(input_ids, local_positions, patches, patch_positions, input_types, attention_mask)
        labels = input_ids.clone()
        # All labels set to -100 are ignored (masked), the loss is only computed for labels in
        # [0, ..., config.vocab_size]
        labels[loss_mask == 0] = -100
        return self.model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
