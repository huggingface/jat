from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn
from transformers import GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput

from gia2.config import Gia2Config
from gia2.image_model import ImageDecoder, ImageEncoder
from gia2.utils import compute_ce_loss, compute_mse_loss, cyclic_expand_dim


class DualBatchReshapeWrapper(nn.Module):
    """
    Wrapper to apply a module to a batch of sequences of batches.

    Args:
        module (`nn.Module`):
            Module to apply to each batch.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: FloatTensor) -> FloatTensor:
        n1, n2 = x.shape[:2]
        x = x.view(n1 * n2, *x.shape[2:])
        x = self.module(x)
        x = x.view(n1, n2, *x.shape[1:])
        return x


@dataclass
class Gia2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_loss: Optional[FloatTensor] = None
    action_loss: Optional[FloatTensor] = None
    pred_observations: torch.FloatTensor = None
    pred_actions: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Gia2Model(GPTNeoPreTrainedModel):
    config_class = Gia2Config

    def __init__(self, config: Gia2Config) -> None:
        super().__init__(config)

        # Encoders
        self.continuous_encoder = nn.Linear(config.max_continuous_size, config.hidden_size)
        self.discrete_encoder = nn.Embedding(18, config.hidden_size)
        self.image_encoder = DualBatchReshapeWrapper(ImageEncoder(config.hidden_size))

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.continuous_decoder = nn.Linear(config.hidden_size, config.max_continuous_size, bias=False)
        self.discrete_decoder = nn.Linear(config.hidden_size, 18, bias=False)
        self.image_decoder = DualBatchReshapeWrapper(ImageDecoder(config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None,
        attention_mask: Optional[BoolTensor] = None,
        token_type_ids: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_loss: bool = True,
        loss_weight: Optional[FloatTensor] = None,
    ) -> Gia2Output:
        """
        Forward pass of the Gia2 model.

        Args:
            input_ids (LongTensor, optional): Input sequence ids. Defaults to None.
            continuous_observations (FloatTensor, optional): Continuous observations. Defaults to None.
            discrete_observations (LongTensor, optional): Discrete observations. Defaults to None.
            image_observations (FloatTensor, optional): Image observations. Defaults to None.
            continuous_actions (FloatTensor, optional): Continuous actions. Defaults to None.
            discrete_actions (LongTensor, optional): Discrete actions. Defaults to None.
            rewards (FloatTensor, optional): Rewards. Defaults to None.
            past_key_values (Tuple[Tuple[FloatTensor]], optional): Past key values. Defaults to None.
            attention_mask (BoolTensor, optional): Attention mask. Defaults to None.
            token_type_ids (LongTensor, optional): Token type ids. Defaults to None.
            position_ids (LongTensor, optional): Position ids. Defaults to None.
            inputs_embeds (FloatTensor, optional): Input embeddings. Defaults to None.
            use_cache (bool, optional): Use cache. Defaults to None.
            output_attentions (bool, optional): Output attentions. Defaults to None.
            output_hidden_states (bool, optional): Output hidden states. Defaults to None.
            return_dict (bool, optional): Return dictionary. Defaults to None.
            return_loss (bool, optional): Return loss. Defaults to True.
            loss_weight (FloatTensor, optional): Loss weight. Defaults to None.

        Returns:
            Gia2Output: Output of the Gia2 model.
        """
        # Encode continuous observations
        if continuous_observations is not None:
            batch_size, seq_len, obs_size = continuous_observations.shape
            continuous_observations = cyclic_expand_dim(continuous_observations, self.config.max_continuous_size)
            inputs_embeds_observations = self.continuous_encoder(continuous_observations)
        # Encode discrete observations
        elif discrete_observations is not None:
            raise NotImplementedError
        # Encode image observations
        elif image_observations is not None:
            inputs_embeds_observations = self.image_encoder(image_observations)
        else:
            inputs_embeds_observations = None

        # Encode continuous actions
        if continuous_actions is not None:
            batch_size, seq_len, action_size = continuous_actions.shape
            continuous_actions = cyclic_expand_dim(continuous_actions, self.config.max_continuous_size)
            inputs_embeds_actions = self.continuous_encoder(continuous_actions)
        # Encode discrete actions
        elif discrete_actions is not None:
            batch_size, seq_len = discrete_actions.shape
            inputs_embeds_actions = self.discrete_encoder(discrete_actions)
        else:
            inputs_embeds_actions = None

        # Concatenate observations and actions
        if inputs_embeds_observations is not None and inputs_embeds_actions is not None:
            inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
                batch_size, 2 * seq_len, self.config.hidden_size
            )
            if attention_mask is not None:
                attention_mask = torch.repeat_interleave(attention_mask, repeats=2, dim=1)

        # Pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # The following will be overwritten if needed
        lm_loss, observation_loss, action_loss = None, None, None
        lm_logits, pred_observations, pred_actions = None, None, None

        # Decode language model TODO: @Clément implement this
        if input_ids is not None:
            lm_logits = torch.randn(
                input_ids.shape[0], input_ids.shape[1], self.config.vocab_size, device=input_ids.device
            )
            if return_loss:
                lm_loss = torch.randn(1, device=input_ids.device)

        # Decode continuous observations
        if continuous_observations is not None:
            pred_observations = self.continuous_decoder(hidden_states[:, 1::2])
            if return_loss:
                observation_loss = compute_mse_loss(
                    pred_observations[:, :-1],
                    continuous_observations[:, 1:],
                    attention_mask[:, 1:],
                    weights=loss_weight,
                )
            pred_observations = pred_observations[..., :obs_size]
        # Decode discrete observations
        elif discrete_observations is not None:
            raise NotImplementedError
        # Decode image observations
        elif image_observations is not None:
            pred_observations = self.image_decoder(hidden_states[:, 1::2])
            if return_loss:
                observation_loss = compute_mse_loss(
                    pred_observations[:, :-1], image_observations[:, 1:], attention_mask[:, 1:], weights=loss_weight
                )

        # Decode continuous actions
        if continuous_actions is not None:
            pred_actions = self.continuous_decoder(hidden_states[:, ::2])
            if return_loss:
                action_loss = compute_mse_loss(pred_actions, continuous_actions, attention_mask, weights=loss_weight)
            pred_actions = pred_actions[..., :action_size]
        # Decode discrete actions
        elif discrete_actions is not None:
            pred_actions = self.discrete_decoder(hidden_states[:, ::2])
            if return_loss:
                action_loss = compute_ce_loss(pred_actions, discrete_actions, attention_mask, weights=loss_weight)

        # Return output
        if return_loss:
            if observation_loss is not None and action_loss is not None:
                loss = 0.0 * observation_loss + 1.0 * action_loss
            elif lm_loss is not None:
                loss = lm_loss
            else:
                raise RuntimeError("No loss to return")
            return Gia2Output(
                loss=loss,
                observation_loss=observation_loss,
                action_loss=action_loss,
                logits=lm_logits,
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            return Gia2Output(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

    @torch.no_grad()
    def get_next_action(
        self,
        continuous_observations: Optional[List[List[float]]] = None,
        discrete_observations: Optional[List[List[int]]] = None,
        image_observations: Optional[List[np.ndarray]] = None,
        continuous_actions: Optional[List[List[float]]] = None,
        discrete_actions: Optional[List[int]] = None,
        rewards: Optional[List[float]] = None,
        deterministic: bool = False,
        action_size: Optional[int] = None,
    ):
        # Get the maximum sequence length
        max_seq_len = self.config.max_position_embeddings // 2

        # Convert to tensors, move to device, and add batch dimension
        if continuous_observations is not None:
            continuous_observations = np.array(continuous_observations, dtype=np.float32)
            continuous_observations = torch.from_numpy(continuous_observations).to(self.device)
            continuous_observations = continuous_observations[None, -max_seq_len:]

        if discrete_observations is not None:
            discrete_observations = np.array(discrete_observations, dtype=np.long)
            discrete_observations = torch.from_numpy(discrete_observations).to(self.device)
            discrete_observations = discrete_observations[None, -max_seq_len:]

        if image_observations is not None:
            image_observations = np.array(image_observations, dtype=np.float32)
            image_observations = torch.from_numpy(image_observations).to(self.device)
            image_observations = torch.permute(image_observations, (0, 3, 1, 2)) / 255.0
            image_observations = (image_observations - 0.5) / 0.5
            image_observations = image_observations[None, -max_seq_len:]

        # For the action, we need to add a fake action to the end of the sequence
        if continuous_actions is not None:
            if len(continuous_actions) == 0:
                continuous_actions = torch.zeros((1, action_size), dtype=torch.float32, device=self.device)
            else:
                continuous_actions = np.array(continuous_actions, dtype=np.float32)
                continuous_actions = torch.from_numpy(continuous_actions).to(self.device)
                last_action = torch.zeros_like(continuous_actions[-1:])
                continuous_actions = torch.cat((continuous_actions, last_action), dim=0)
            continuous_actions = continuous_actions[None, -max_seq_len:]

        if discrete_actions is not None:
            if len(discrete_actions) == 0:
                discrete_actions = torch.zeros((1,), dtype=torch.long, device=self.device)
            else:
                discrete_actions = np.array(discrete_actions, dtype=np.int64)
                discrete_actions = torch.from_numpy(discrete_actions).to(self.device)
                last_action = torch.zeros_like(discrete_actions[-1:])
                discrete_actions = torch.cat((discrete_actions, last_action), dim=0)
            discrete_actions = discrete_actions[None, -max_seq_len:]

        if rewards is not None:
            rewards = np.array(rewards, dtype=np.float32)
            rewards = torch.from_numpy(rewards).to(self.device)
            rewards = rewards[None, -max_seq_len:]

        outputs = self(
            continuous_observations=continuous_observations,
            discrete_observations=discrete_observations,
            image_observations=image_observations,
            continuous_actions=continuous_actions,
            discrete_actions=discrete_actions,
            rewards=rewards,
            return_loss=False,
        )

        if continuous_actions is not None:
            return outputs.pred_actions[0, -1].cpu().numpy()
        elif discrete_actions is not None:
            logits = outputs.pred_actions[0, -1, :action_size]
            if deterministic:
                return logits.argmax().cpu().numpy()
            else:  # sample
                return torch.multinomial(logits.softmax(dim=-1), num_samples=1)[0].item()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs
