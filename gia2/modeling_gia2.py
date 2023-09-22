from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from transformers import GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_gia2 import Gia2Config


def compute_mse_loss(
    predicted: FloatTensor, true: FloatTensor, mask: BoolTensor, weights: FloatTensor = None
) -> FloatTensor:
    """
    Compute the Mean Squared Error (MSE) loss between predicted and true observations, considering valid timesteps.

    Args:
        predicted (`FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Predicted observations at the output of the model.
        true (`FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Ground truth observations.
        mask (`BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.

    Returns:
        loss (`FloatTensor` of shape `(,)`):
            MSE loss between predicted and true observations.
    """
    # Compute element-wise MSE loss
    loss = F.mse_loss(predicted, true, reduction="none")

    # Average the loss over all dimensions after the second one
    for dim in reversed(range(2, loss.dim())):
        loss = loss.mean(dim=dim)

    # Use the mask to zero out invalid entries
    loss = torch.sum(loss * mask, dim=1)

    # Apply weights if provided
    if weights is not None:
        loss = loss * weights

    # Sum the loss and normalize by the number of valid elements
    loss = loss.sum() / mask.sum()

    return loss


def compute_ce_loss(
    predicted: FloatTensor, true: torch.LongTensor, mask: BoolTensor, weights: FloatTensor = None
) -> FloatTensor:
    """
    Compute the Cross Entropy (CE) loss between predicted logits and true class labels, considering valid timesteps.

    Args:
        predicted (`FloatTensor` of shape `(batch_size, max_seq_len, num_classes)`):
            Predicted logits at the output of the model.
        true (`torch.LongTensor` of shape `(batch_size, max_seq_len)`):
            Ground truth class labels.
        mask (`BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.

    Returns:
        loss (`FloatTensor` of shape `(,)`):
            CE loss between predicted logits and true class labels.
    """

    # Compute element-wise CE loss
    loss = F.cross_entropy(predicted.view(-1, predicted.size(-1)), true.view(-1), reduction="none")
    loss = loss.view(true.size())

    # Use the mask to zero out invalid entries
    loss = torch.sum(loss * mask, dim=1)

    # Apply weights if provided
    if weights is not None:
        loss = loss * weights

    # Sum the loss and normalize by the number of valid elements
    loss = loss.sum() / mask.sum()

    return loss


def cyclic_expand_dim(tensor: Tensor, expanded_dim_size: int) -> Tensor:
    """
    Expands the last dimension of a tensor cyclically to a specified size.

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, seq_len, ...)`):
            Input tensor whose last dimension is to be expanded cyclically.
        expanded_dim_size (`int`):
            The desired size of the last dimension after expansion.

    Returns:
        `torch.Tensor` of shape `(batch_size, seq_len, expanded_dim_size)`:
            A tensor with its last dimension expanded cyclically to the specified size.

    Examples:
        >>> tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> cyclic_expand_dim(tensor, 5)
        tensor([[[1, 2, 1, 2, 1], [3, 4, 3, 4, 3]], [[5, 6, 5, 6, 5], [7, 8, 7, 8, 7]]])
    """
    B, L, X = tensor.shape
    if expanded_dim_size < X:
        raise ValueError(
            f"Expanded dimension size ({expanded_dim_size}) must be greater than the original dimension size ({X})."
        )
    indices = torch.arange(expanded_dim_size) % X
    return tensor[..., indices]


class ResidualBlock(nn.Module):
    """
    A residual block module that consists of two convolutional layers with a residual connection.

    Args:
        in_shape (`Tuple[int, int, int]`):
            Shape of the input tensor.
        out_channels (`int`):
            Number of output channels.

    Returns:
        `torch.Tensor` of shape `(batch_size, out_channels, in_shape[1], in_shape[2])`:
            Output tensor.
    """

    def __init__(self, in_shape: Tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        out_shape = (out_channels, in_shape[1], in_shape[2])

        self.conv1 = nn.Conv2d(in_shape[0], out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(out_shape)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.LayerNorm(out_shape)

        # Handling the change in dimensions with a 1x1 convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_shape[0], out_channels, kernel_size=1, stride=1), nn.LayerNorm(out_shape)
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = F.leaky_relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out)


class AttentionLayer(nn.Module):
    """
    Attention layer that applies an attention mechanism to the input tensor.

    Args:
        num_channels (`int`):
            Number of channels.

    Returns:
        `torch.Tensor`:
            Output tensor of the same shape as the input tensor.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 8, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImageEncoder(nn.Module):
    """
    Image encoder that encodes a batch of images.

    Args:
        hidden_size (`int`):
            Size of the output hidden state.

    Returns:
        `torch.Tensor` of shape `(batch_size, hidden_size)`:
            Output tensor.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)  # 42x42
        self.norm1 = nn.InstanceNorm2d(32)
        self.att1 = AttentionLayer(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 21x21
        self.norm2 = nn.InstanceNorm2d(64)
        self.att2 = AttentionLayer(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 11x11
        self.norm3 = nn.InstanceNorm2d(128)
        self.att3 = AttentionLayer(128)
        self.fc = nn.Linear(128 * 11 * 11, hidden_size)  # Adjusted to the new spatial dimension

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = self.att1(x)
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = self.att2(x)
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = self.att3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    """
    Image decoder that decodes a batch of encoded representations.

    Args:
        hidden_size (`int`):
            Size of the input hidden state.

    Returns:
        `torch.Tensor` of shape `(batch_size, 4, 84, 84)`:
            Output tensor representing the reconstructed images.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, 128 * 11 * 11)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 21x21
        self.norm1 = nn.InstanceNorm2d(64)
        self.att1 = AttentionLayer(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # 42x42
        self.norm2 = nn.InstanceNorm2d(32)
        self.att2 = AttentionLayer(32)
        self.deconv3 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1)  # 84x84

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.fc(x)
        x = x.view(x.size(0), 128, 11, 11)  # Reshape to the spatial dimension of encoder's last conv layer
        x = F.leaky_relu(self.norm1(self.deconv1(x)))  # 22x22
        x = F.interpolate(x, size=(21, 21))  # 21x21
        x = self.att1(x)
        x = F.leaky_relu(self.norm2(self.deconv2(x)))
        x = self.att2(x)
        x = F.tanh(self.deconv3(x))
        return x


class DualBatchReshapeWrapper(nn.Module):
    """
    Wrapper to make a module designed for a single batch work with a dual batch.

    Args:
        module (`nn.Module`):
            Module to be wrapped.
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
    """
    Output of the Gia2 model.

    The model can be used for both RL and NLP tasks. For RL tasks, the model takes in observations and actions
    (`continuous_observations`, `discrete_actions`, etc.). For textual tasks, the model takes in a sequence of tokens
    and/or images (`input_ids`, `image`). The output depends on the type of input.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            For RL input, the loss is the sum of the observation loss and the action loss.
            For textual input, the causal language modeling loss.
        observation_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Only returned when RL input is provided. The MSE loss between predicted and true observations for
            continuous observations and the cross-entropy loss for discrete observations.
        action_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Only returned when RL input is provided. The MSE loss between predicted and true actions for
            continuous actions and the cross-entropy loss for discrete actions.
        pred_observations (`torch.FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Only returned when RL input is provided. Predicted observations from t=1 to t=max_seq_len+1.
        pred_actions (`torch.FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Only returned when RL input is provided. Predicted actions from t=0 to t=max_seq_len. When input actions
            are discrete, the predicted actions are logits.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[FloatTensor] = None
    observation_loss: Optional[FloatTensor] = None
    action_loss: Optional[FloatTensor] = None
    pred_observations: Optional[FloatTensor] = None
    pred_actions: Optional[FloatTensor] = None
    logits: Optional[FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None
    hidden_states: Optional[Tuple[FloatTensor]] = None
    attentions: Optional[Tuple[FloatTensor]] = None


class Gia2Model(GPTNeoPreTrainedModel):
    """
    Gia2 model.
    """

    config_class = Gia2Config

    def __init__(self, config: Gia2Config) -> None:
        super().__init__(config)

        # Encoders
        self.continuous_encoder = nn.Linear(config.max_continuous_size, config.hidden_size)
        self.discrete_encoder = nn.Embedding(config.max_discrete_value, config.hidden_size)
        self.image_encoder = DualBatchReshapeWrapper(ImageEncoder(config.hidden_size))

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.continuous_decoder = nn.Linear(config.hidden_size, config.max_continuous_size, bias=False)
        self.discrete_decoder = nn.Linear(config.hidden_size, config.max_discrete_value, bias=False)
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

    # copied from gpt-neo, allows to use .generate()
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


Gia2Model.register_for_auto_class("AutoModelForCausalLM")