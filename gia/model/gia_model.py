from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from gia import GiaConfig

from .embedding import Embeddings


class GiaModel(nn.Module):
    """
    GiaModel is a wrapper around a transformer model that takes in both text and image patches as input.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GiaConfig`]):
            Model configuration class with all the parameters of the model.
    """

    def __init__(self, config: GiaConfig) -> None:
        super().__init__()
        if config.use_pretrained:
            self.causal_lm_model = AutoModelForCausalLM.from_pretrained(
                config.causal_lm_name, config=config.causal_lm_config
            )
        else:
            self.causal_lm_model = AutoModelForCausalLM.from_config(config.causal_lm_config)

        self.emb = Embeddings(
            config.embed_dim,
            config.vocab_size,
            config.max_local_position,
            config.patch_size,
            config.image_vocab_size,
            config.num_res_channels,
            config.num_groups,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        patches: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        input_types: Optional[torch.LongTensor] = None,
        local_positions: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        loss_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        use_cache=False,
    ) -> CausalLMOutputWithPast:
        """
        Run a forward pass through the model. Takes in several inputs and returns a `CausalLMOutputWithPast` object.
        return self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values,

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            patches (`torch.Tensor` of shape `(batch_size, sequence_length, patch_size, patch_size)`, *optional*):
                Tensor containing the image patch data for each image patch in the sequence.
            patch_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Tensor containing the position of each image patch in the sequence.
            input_types (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Tensor indicating the type of each input in the sequence (0 for tokens and 1 for image patches).
            local_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of local positions of each token in the sequence.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[False, True]`:

                    - True for tokens that are **not masked**,
                    - False for tokens that are **masked**.

            loss_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to ignore certain tokens in the loss computation. Mask values selected in `{True, False}`:

                    - True for tokens that are **not ignored**,
                    - False for tokens that are **ignored**.

        Returns:
            `CausalLMOutputWithPast`: Output object from `transformers.ModelOutputs`.

        Raises:
            ValueError: If one of the following conditions is met:
                - both input_ids and patches are None
                - input_ids and patches are provided but input_types is None
                - the input_types tensor contains values other than 0 or 1
                - patches is provided but patch_positions is None (and vice-versa)
        """
        embeds = self.emb(input_ids, patches, patch_positions, input_types, local_positions, attention_mask)
        if input_ids is not None:
            labels = input_ids.clone()
            # All labels set to -100 are ignored (masked), the loss is only computed for labels in
            # [0, ..., config.vocab_size]
            if loss_mask is not None:
                labels[~loss_mask] = -100
        else:
            labels = None
        return self.causal_lm_model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        patches: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        input_types: Optional[torch.LongTensor] = None,
        local_positions: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        num_tokens: int = 1,
    ):
        raise NotImplementedError("GiaModel.generate is not implemented yet.")
