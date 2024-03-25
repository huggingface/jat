import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from transformers import GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings

from .configuration_jat import JatConfig
from .processing_jat import JatProcessor


def compute_mse_loss(
    predicted: FloatTensor, true: FloatTensor, mask: Optional[BoolTensor], weights: Optional[FloatTensor] = None
) -> FloatTensor:
    """
    Compute the Mean Squared Error (MSE) loss between predicted and true observations, considering valid timesteps.

    Args:
        predicted (`FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Predicted observations at the output of the model.
        true (`FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Ground truth observations.
        mask (`BoolTensor` of shape `(batch_size, max_seq_len)`, *optional*):
            Boolean mask indicating valid timesteps.
        weights (`FloatTensor` of shape `(batch_size, max_seq_len)`, *optional*):
            Weights to be applied to the loss.

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
    if mask is not None:
        loss = loss * mask

    # Apply weights if provided
    if weights is not None:
        loss = loss * weights

    # Sum the loss and normalize by the number of valid elements
    loss = loss.sum() / mask.sum() if mask is not None else loss.mean()

    return loss


def compute_ce_loss(
    logits: FloatTensor, labels: torch.LongTensor, mask: Optional[BoolTensor], weights: Optional[FloatTensor] = None
) -> FloatTensor:
    """
    Compute the Cross Entropy (CE) loss between predicted logits and true class labels, considering valid timesteps.

    Args:
        logits (`FloatTensor` of shape `(batch_size, max_seq_len, [inner_size,] num_classes)`):
            Predicted logits at the output of the model.
        labels (`torch.LongTensor` of shape `(batch_size, max_seq_len, [inner_size,])`):
            Ground truth class labels.
        mask (`BoolTensor` of shape `(batch_size, max_seq_len)`, *optional*):
            Boolean mask indicating valid timesteps.
        weights (`FloatTensor` of shape `(batch_size, max_seq_len)`, *optional*):
            Weights to be applied to the loss.

    Returns:
        loss (`FloatTensor` of shape `(,)`):
            CE loss between predicted logits and true class labels.
    """
    if mask is not None:
        logits = logits[mask.bool()]  # (Y, X, C)
        labels = labels[mask.bool()]  # (Y, X)
        if weights is not None:
            weights = weights[mask.bool()]  # (Y,)
    else:
        logits = logits.flatten(end_dim=2)  # (B, L, X, C) -> (B*L, X, C)
        labels = labels.flatten(end_dim=1)  # (B, L, X) -> (B*L, X)
        if weights is not None:
            weights = weights.flatten(end_dim=1)  # (B, L) -> (B*L,)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")  # (Y*X,)
    loss = loss.view(labels.size())  # (Y, X)
    loss = loss.mean(-1)  # (Y,)

    # Multiply the loss by the weights
    if weights is not None:
        loss = loss * weights  # (Y,)

    # Average the loss
    loss = loss.mean()

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
        return F.leaky_relu(out, inplace=True)


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
        x = F.leaky_relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.att1(x)
        x = F.leaky_relu(self.norm2(self.conv2(x)), inplace=True)
        x = self.att2(x)
        x = F.leaky_relu(self.norm3(self.conv3(x)), inplace=True)
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
        x = F.leaky_relu(self.norm1(self.deconv1(x)), inplace=True)  # 22x22
        x = F.interpolate(x, size=(21, 21))  # 21x21
        x = self.att1(x)
        x = F.leaky_relu(self.norm2(self.deconv2(x)), inplace=True)
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
class JatOutput(ModelOutput):
    """
    Output of the Jat model.

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
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`):
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


class JatModel(GPTNeoPreTrainedModel):
    """
    Jat model.
    """

    config_class = JatConfig

    def __init__(self, config: JatConfig) -> None:
        super().__init__(config)

        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        max_discrete_value = config.max_discrete_value
        max_continuous_size = config.max_continuous_size
        self.observation_loss_coef = config.observation_loss_coef
        self.action_loss_coef = config.action_loss_coef

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Encoders
        self.vit_encoder = ViTPatchEmbeddings(config)
        self.single_discrete_encoder = self.transformer.wte
        self.continuous_encoder = nn.Linear(max_continuous_size, hidden_size)
        self.multi_discrete_encoder = nn.Sequential(
            self.single_discrete_encoder,  # (B, L, X, H)
            nn.Linear(hidden_size, hidden_size // 50),  # (B, L, X, H // 50)
            nn.ReLU(),
            nn.Flatten(start_dim=2),  # (B, L, X * (H // 50))
            nn.Linear(max_discrete_value * (hidden_size // 50), hidden_size - 1),  # (B, L, H)
        )  # -1 to account for the reward
        self.image_encoder = DualBatchReshapeWrapper(ImageEncoder(hidden_size))

        # Decoders
        self.single_discrete_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.continuous_decoder = nn.Linear(hidden_size, max_continuous_size)
        self.multi_discrete_decoder = nn.Sequential(
            nn.Linear(hidden_size, max_discrete_value * (hidden_size // 50)),  # (B, L, X * (H // 50))
            nn.Unflatten(dim=2, unflattened_size=(max_discrete_value, hidden_size // 50)),  # (B, L, X, H // 50)
            nn.ReLU(),
            nn.Linear(hidden_size // 50, hidden_size),  # (B, L, X, H)
            nn.ReLU(),
            nn.Linear(hidden_size, 8, bias=False),  # (B, L, X, 8) - the max possible value in the dataset is 8
        )
        self.image_decoder = DualBatchReshapeWrapper(ImageDecoder(hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def embed_textual(
        self,
        input_ids: Optional[LongTensor],
        pixel_values: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        text_inputs_embeds = self.single_discrete_encoder(input_ids) if input_ids is not None else None
        image_inputs_embeds = self.vit_encoder(pixel_values) if pixel_values is not None else None
        # Concatenate text and image inputs
        if image_inputs_embeds is not None and text_inputs_embeds is not None:
            inputs_embeds = torch.cat((image_inputs_embeds, text_inputs_embeds), dim=1)
            # Add attention mask for image inputs
            image_mask = torch.ones(image_inputs_embeds.shape[:2], dtype=torch.bool, device=self.device)
            if attention_mask is None:
                attention_mask = torch.ones(text_inputs_embeds.shape[:2], dtype=torch.bool, device=self.device)
            attention_mask = torch.cat((image_mask, attention_mask), dim=1)
        elif image_inputs_embeds is not None:
            inputs_embeds = image_inputs_embeds
        elif text_inputs_embeds is not None:
            inputs_embeds = text_inputs_embeds
            attention_mask = attention_mask
        else:
            raise ValueError("At least one of `input_ids` or `pixel_values` must be provided.")
        return inputs_embeds, attention_mask

    def embed_rl(
        self,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
    ):
        # Prepare RL inputs (pad and cat rewards to observations)
        assert rewards is not None
        if continuous_observations is not None:
            continuous_observations = torch.cat((continuous_observations, rewards.unsqueeze(-1)), dim=-1)
            continuous_observations = cyclic_expand_dim(continuous_observations, self.config.max_continuous_size)
        if continuous_actions is not None:
            continuous_actions = cyclic_expand_dim(continuous_actions, self.config.max_continuous_size)

        # Encode
        if continuous_observations is not None:
            batch_size, seq_len = continuous_observations.shape[:2]
            inputs_embeds_observations = self.continuous_encoder(continuous_observations)
        elif discrete_observations is not None:
            batch_size, seq_len = discrete_observations.shape[:2]
            inputs_embeds_observations = self.multi_discrete_encoder(discrete_observations)
            inputs_embeds_observations = torch.cat((inputs_embeds_observations, rewards.unsqueeze(-1)), dim=-1)
        elif image_observations is not None:
            batch_size, seq_len = image_observations.shape[:2]
            inputs_embeds_observations = self.image_encoder(image_observations)
        else:
            raise ValueError("Missing observations.")
        if continuous_actions is not None:
            inputs_embeds_actions = self.continuous_encoder(continuous_actions)
        elif discrete_actions is not None:
            inputs_embeds_actions = self.single_discrete_encoder(discrete_actions)
        else:
            raise ValueError("Missing actions.")

        # Concatenate observations and actions
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2)
        inputs_embeds = inputs_embeds.view(batch_size, 2 * seq_len, self.config.hidden_size)
        if attention_mask is not None:
            attention_mask = torch.repeat_interleave(attention_mask, repeats=2, dim=1)
        return inputs_embeds, attention_mask

    def output_textual(
        self,
        transformer_outputs,
        input_ids: Optional[LongTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ):
        hidden_states = transformer_outputs[0]
        loss = None
        # Get only textual hidden states
        lm_logits = self.single_discrete_decoder(hidden_states)
        if return_loss:
            if input_ids is None:
                raise ValueError("Input IDs must be provided when `return_loss=True`.")

            # Shift so that tokens < n predict n
            num_text_tokens = input_ids.shape[1]
            shift_logits = lm_logits[:, -num_text_tokens:-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -num_text_tokens:]
                shift_attention_mask = shift_attention_mask[:, 1:]
            else:
                shift_attention_mask = torch.ones(shift_labels.shape, dtype=bool, device=self.device)
            shift_logits = shift_logits[shift_attention_mask.bool()]
            shift_labels = shift_labels[shift_attention_mask.bool()]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return JatOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def output_rl(
        self,
        transformer_outputs,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        loss_weight: Optional[FloatTensor] = None,
    ):
        hidden_states = transformer_outputs.last_hidden_state
        loss, observation_loss, action_loss = None, None, None
        # Observations
        assert rewards is not None
        observations_mask = attention_mask[:, 1::2] if attention_mask is not None else None
        if continuous_observations is not None:
            if self.observation_loss_coef == 0.0:
                warnings.warn("observation_loss_coef is 0.0, skipping memory-intensive observations prediction.")
                pred_observations = None
                observation_loss = 0.0
            else:
                obs_size = continuous_observations.shape[-1]
                continuous_observations = torch.cat((continuous_observations, rewards.unsqueeze(-1)), dim=-1)
                continuous_observations = cyclic_expand_dim(continuous_observations, self.config.max_continuous_size)
                pred_observations = self.continuous_decoder(hidden_states[:, 1::2])
                if return_loss:
                    observation_loss = compute_mse_loss(
                        pred_observations[:, :-1],
                        continuous_observations[:, 1:],
                        observations_mask[:, 1:] if observations_mask is not None else None,
                        weights=loss_weight[:, 1:] if loss_weight is not None else None,
                    )
                pred_observations = pred_observations[..., :obs_size]
        elif discrete_observations is not None:  # Note: reward is not predicted
            if self.observation_loss_coef == 0.0:
                warnings.warn("observation_loss_coef is 0.0, skipping memory-intensive observations prediction.")
                pred_observations = None
                observation_loss = 0.0
            else:
                warnings.warn("Discrete observations prediction are not supported yet.")  # way too expensive
                pred_observations = None
                observation_loss = 0.0
                # pred_observations = self.multi_discrete_decoder(hidden_states[:, 1::2])
                # if return_loss:
                #     observation_loss = compute_ce_loss(
                #         pred_observations[:, :-1],
                #         discrete_observations[:, 1:],
                #         observations_mask[:, 1:] if observations_mask is not None else None,
                #         weights=loss_weight[:, 1:] if loss_weight is not None else None,
                #     )
        elif image_observations is not None:
            if self.observation_loss_coef == 0.0:
                warnings.warn("observation_loss_coef is 0.0, skipping memory-intensive observations prediction.")
                pred_observations = None
                observation_loss = 0.0
            else:
                pred_observations = self.image_decoder(hidden_states[:, 1::2])
                if return_loss:
                    observation_loss = compute_mse_loss(
                        pred_observations[:, :-1],
                        image_observations[:, 1:],
                        observations_mask[:, 1:] if observations_mask is not None else None,
                        weights=loss_weight[:, 1:] if loss_weight is not None else None,
                    )

        # Actions
        actions_mask = attention_mask[:, ::2] if attention_mask is not None else None
        if continuous_actions is not None:
            act_size = continuous_actions.shape[-1]
            continuous_actions = cyclic_expand_dim(continuous_actions, self.config.max_continuous_size)
            pred_actions = self.continuous_decoder(hidden_states[:, ::2])
            if return_loss:
                action_loss = compute_mse_loss(pred_actions, continuous_actions, actions_mask, weights=loss_weight)
            pred_actions = pred_actions[..., :act_size]
        elif discrete_actions is not None:
            pred_actions = self.single_discrete_decoder(hidden_states[:, ::2])
            if return_loss:
                action_loss = compute_ce_loss(pred_actions, discrete_actions, actions_mask, weights=loss_weight)

        # Return output
        if return_loss:
            loss = self.observation_loss_coef * observation_loss + self.action_loss_coef * action_loss

        if not return_dict:
            output = (pred_observations, pred_actions) + transformer_outputs[1:]
            return ((loss, observation_loss, action_loss) + output) if loss is not None else output

        return JatOutput(
            loss=loss,
            observation_loss=observation_loss,
            action_loss=action_loss,
            pred_observations=pred_observations,
            pred_actions=pred_actions,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        pixel_values: Optional[FloatTensor] = None,
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
        return_loss: bool = True,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_weight: Optional[FloatTensor] = None,
    ) -> JatOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Textual tasks
        if input_ids is not None or pixel_values is not None:
            inputs_embeds, attention_mask = self.embed_textual(input_ids, pixel_values, attention_mask)
        # RL tasks
        elif (
            continuous_observations is not None or discrete_observations is not None or image_observations is not None
        ):
            inputs_embeds, attention_mask = self.embed_rl(
                continuous_observations,
                discrete_observations,
                image_observations,
                continuous_actions,
                discrete_actions,
                rewards,
                attention_mask,
            )
        else:
            raise ValueError("Input not provided.")

        # Pass through transformer
        transformer_outputs = self.transformer(
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

        if input_ids is not None or pixel_values is not None:
            return self.output_textual(transformer_outputs, input_ids, attention_mask, return_loss, return_dict)
        else:
            return self.output_rl(
                transformer_outputs,
                continuous_observations,
                discrete_observations,
                image_observations,
                continuous_actions,
                discrete_actions,
                rewards,
                attention_mask,
                return_loss,
                return_dict,
                loss_weight,
            )

    def reset_rl(self):
        self._last_key_values = None
        self.last_discrete_observation = None
        self.last_continuous_observation = None
        self.last_text_observation = None
        self.last_image_observation = None
        self.last_discrete_action = None
        self.last_continuous_action = None
        self.last_reward = None

    @torch.no_grad()
    def get_next_action(
        self,
        processor: JatProcessor,
        continuous_observation: Optional[List[float]] = None,
        discrete_observation: Optional[List[int]] = None,
        text_observation: Optional[str] = None,
        image_observation: Optional[np.ndarray] = None,
        action_space: Union[spaces.Box, spaces.Discrete] = None,
        reward: Optional[float] = None,
        deterministic: bool = False,
    ):
        # Get the maximum sequence length
        max_length = self.config.max_position_embeddings // 2

        # Convert everything to lists
        def to_list(x):
            return x.tolist() if isinstance(x, np.ndarray) else x

        continuous_observation = to_list(continuous_observation)
        discrete_observation = to_list(discrete_observation)

        # Add a fake action to the end of the sequence
        if isinstance(action_space, spaces.Box):
            fake_continuous_action = [0.0 for _ in range(action_space.shape[0])]
            fake_discrete_action = None
        elif isinstance(action_space, spaces.Discrete):
            fake_continuous_action = None
            fake_discrete_action = 0

        continuous_observations = [continuous_observation] if continuous_observation is not None else None
        discrete_observations = [discrete_observation] if discrete_observation is not None else None
        text_observations = [text_observation] if text_observation is not None else None
        image_observations = [image_observation] if image_observation is not None else None
        continuous_actions = [fake_continuous_action] if fake_continuous_action is not None else None
        discrete_actions = [fake_discrete_action] if fake_discrete_action is not None else None
        rewards = [reward] if reward is not None else [0.0]

        if self._last_key_values is not None:
            # We concatenate the last observation with the current one
            continuous_observations = (
                [self.last_continuous_observation] + continuous_observations
                if continuous_observations is not None
                else None
            )
            discrete_observations = (
                [self.last_discrete_observation] + discrete_observations if discrete_observations is not None else None
            )
            text_observations = (
                [self.last_text_observation] + text_observations if text_observations is not None else None
            )
            image_observations = (
                [self.last_image_observation] + image_observations if image_observations is not None else None
            )
            continuous_actions = (
                [self.last_continuous_action] + continuous_actions if continuous_actions is not None else None
            )
            discrete_actions = [self.last_discrete_action] + discrete_actions if discrete_actions is not None else None
            rewards = [self.last_reward] + rewards

        # Store the last observation
        self.last_continuous_observation = continuous_observations[-1] if continuous_observations is not None else None
        self.last_discrete_observation = discrete_observations[-1] if discrete_observations is not None else None
        self.last_text_observation = text_observations[-1] if text_observations is not None else None
        self.last_image_observation = image_observations[-1] if image_observations is not None else None
        self.last_reward = rewards[-1]

        # Add the batch dimension
        continuous_observations = [continuous_observations] if continuous_observations is not None else None
        discrete_observations = [discrete_observations] if discrete_observations is not None else None
        text_observations = [text_observations] if text_observations is not None else None
        image_observations = [image_observations] if image_observations is not None else None
        continuous_actions = [continuous_actions] if continuous_actions is not None else None
        discrete_actions = [discrete_actions] if discrete_actions is not None else None
        rewards = [rewards]

        # Process the inputs
        processed = processor(
            continuous_observations=continuous_observations,
            discrete_observations=discrete_observations,
            text_observations=text_observations,
            image_observations=image_observations,
            continuous_actions=continuous_actions,
            discrete_actions=discrete_actions,
            rewards=rewards,
            truncation=True,
            truncation_side="left",
            max_length=max_length,
            return_tensors="pt",
        )
        processed.to(self.device)

        # Forward pass
        outputs = self(**processed, past_key_values=self._last_key_values, return_loss=False)

        # Truncate the past key-values
        self._last_key_values = tuple(
            tuple(pkv[:, :, -self.config.max_position_embeddings + 2 :] for pkv in pkvs)
            for pkvs in outputs.past_key_values
        )
        # Store the last key values
        # We remove the last two values, as the inputs are [s_0, 0], [s_0, a_0, s_1, 0], [s_1, a_1, s_2, 0], ...
        self._last_key_values = tuple(tuple(pkv[:, :, :-2] for pkv in pkvs) for pkvs in self._last_key_values)

        # Return the predicted action
        if continuous_actions is not None:
            self.last_continuous_action = outputs.pred_actions[0, -1].cpu().tolist()
            return self.last_continuous_action
        elif discrete_actions is not None:
            logits = outputs.pred_actions[0, -1, : action_space.n]
            if deterministic:
                self.last_discrete_action = logits.argmax().cpu().item()
            else:  # sample
                self.last_discrete_action = torch.multinomial(logits.softmax(dim=-1), num_samples=1)[0].item()
            return self.last_discrete_action

    # Allows to use .generate()
    def prepare_inputs_for_generation(self, input_ids, pixel_values=None, past_key_values=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            pixel_values = None
            input_ids = input_ids[:, -1].unsqueeze(-1)

        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

        return model_inputs


JatModel.register_for_auto_class("AutoModelForCausalLM")
