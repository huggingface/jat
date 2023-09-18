from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor, nn
from transformers import GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput

from gia2.config import Gia2Config
from gia2.utils import compute_ce_loss, compute_mse_loss, cyclic_expand_dim


class ResBlock(nn.Module):
    """
    Residual block with instance normalization.

    Args:
        num_channels (`int`):
            Number of input and output channels.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(num_channels)  # the batch is too self-correlated to use batch norm
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(num_channels)

    def forward(self, x: FloatTensor) -> FloatTensor:
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += residual
        return F.relu(out)


class ImageEncoder(nn.Module):
    """
    Image encoder. Input is a batch of images of shape (N, 4, 84, 84).

    Args:
        hidden_size (`int`):
            Size of the output hidden state.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.res1 = ResBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.res2 = ResBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res3 = ResBlock(128)
        self.fc = nn.Linear(128 * 10 * 10, hidden_size)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.res1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.res2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.res3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    """
    Image decoder. Output is a batch of images of shape (N, 4, 84, 84).

    Args:
        hidden_size (`int`):
            Size of the input hidden state.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, 128 * 10 * 10)
        self.convt1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.res1 = ResBlock(64)
        self.convt2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.res2 = ResBlock(32)
        self.convt3 = nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.fc(x)
        x = x.view(x.size(0), 128, 10, 10)
        x = F.relu(self.convt1(x))
        x = self.res1(x)
        x = F.relu(self.convt2(x))
        x = self.res2(x)
        x = self.convt3(x)
        return x


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
class GIA2Output(ModelOutput):
    pred_observations: torch.FloatTensor = None
    pred_actions: torch.FloatTensor = None
    observation_loss: Optional[FloatTensor] = None
    action_loss: Optional[FloatTensor] = None
    loss: Optional[FloatTensor] = None


class GIA2Model(GPTNeoPreTrainedModel):
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
        self.discrete_decoder = nn.Linear(config.hidden_size, 18)
        self.image_decoder = DualBatchReshapeWrapper(ImageDecoder(config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        return_loss: bool = True,
        loss_weight: Optional[FloatTensor] = None,
    ) -> GIA2Output:
        if continuous_observations is not None:
            batch_size, seq_len, obs_size = continuous_observations.shape
            continuous_observations = cyclic_expand_dim(continuous_observations, self.config.max_continuous_size)
            inputs_embeds_observations = self.continuous_encoder(continuous_observations)
        elif discrete_observations is not None:
            raise NotImplementedError
        elif image_observations is not None:
            norm_image_observations = image_observations.transpose(4, 2) / 128 - 1.0  # to channel first and normalize
            inputs_embeds_observations = self.image_encoder(norm_image_observations)

        if continuous_actions is not None:
            batch_size, seq_len, action_size = continuous_actions.shape
            continuous_actions = cyclic_expand_dim(continuous_actions, self.config.max_continuous_size)
            inputs_embeds_actions = self.continuous_encoder(continuous_actions)
        elif discrete_actions is not None:
            raise NotImplementedError

        # Interleave observations and actions repeat attention_mask accordingly
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )
        if attention_mask is not None:
            input_attention_mask = torch.repeat_interleave(attention_mask, repeats=2, dim=1)
        else:
            input_attention_mask = None

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=input_attention_mask)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[:, 1::2]
        hidden_actions = hidden_states[:, ::2]

        if continuous_observations is not None:
            pred_observations = self.continuous_decoder(hidden_observations)
            if return_loss:
                observation_loss = compute_mse_loss(
                    pred_observations[:, :-1],
                    continuous_observations[:, 1:],
                    attention_mask[:, 1:],
                    weights=loss_weight,
                )
            pred_observations = pred_observations[..., :obs_size]
        elif discrete_observations is not None:
            raise NotImplementedError
        elif image_observations is not None:
            raise NotImplementedError

        if continuous_actions is not None:
            pred_actions = self.continuous_decoder(hidden_actions)
            if return_loss:
                action_loss = compute_mse_loss(pred_actions, continuous_actions, attention_mask, weights=loss_weight)
            pred_actions = pred_actions[..., :action_size]
        elif discrete_actions is not None:
            raise NotImplementedError

        if return_loss:
            return GIA2Output(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=0.0 * observation_loss + 1.0 * action_loss,
            )
        else:
            return GIA2Output(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
            )


class AtariModel(GPTNeoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Encoders
        self.encoder = DualBatchReshapeWrapper(ImageEncoder(self.config.hidden_size))
        self.embedding = nn.Embedding(18, self.config.hidden_size)

        # Transformer
        self.transformer = GPTNeoModel(config)

        # Decoders
        self.image_decoder = DualBatchReshapeWrapper(ImageDecoder(self.config.hidden_size))
        self.logits_decoder = nn.Linear(self.config.hidden_size, 18)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image_observations: Optional[FloatTensor] = None,
        discrete_actions: Optional[FloatTensor] = None,
        rewards: Optional[FloatTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        return_loss: bool = True,
    ):
        # to channel first and normalize
        norm_image_observations = image_observations.transpose(4, 2).transpose(3, 4) / 128 - 1.0
        inputs_embeds_observations = self.encoder(norm_image_observations)
        inputs_embeds_actions = self.embedding(discrete_actions)
        batch_size, seq_len, _ = inputs_embeds_actions.shape

        # Interleave observations and actions
        inputs_embeds = torch.cat((inputs_embeds_observations, inputs_embeds_actions), dim=2).view(
            batch_size, 2 * seq_len, self.config.hidden_size
        )
        if attention_mask is not None:
            _attention_mask = attention_mask.repeat_interleave(2, dim=1)
        else:
            _attention_mask = None

        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=_attention_mask)

        hidden_states = transformer_outputs[0]

        # Un-interleave observations and actions (warning, shifted by 1)
        hidden_observations = hidden_states[..., 1::2, :]
        hidden_actions = hidden_states[..., ::2, :]

        pred_observations = self.image_decoder(hidden_observations)
        pred_actions = self.logits_decoder(hidden_actions)

        observation_loss, action_loss = None, None
        if return_loss:
            obs_criterion = nn.MSELoss(reduction="none")
            raw_loss = obs_criterion(pred_observations[:, :-1], norm_image_observations[:, 1:])  # (N, L-1, C, H, W)
            raw_loss = torch.mean(raw_loss, (2, 3, 4))  # (N, L-1)
            masked_loss = raw_loss * attention_mask[:, 1:]  # (N, L-1)
            observation_loss = torch.sum(masked_loss) / torch.sum(attention_mask[:, 1:])

            action_criterion = nn.CrossEntropyLoss(reduction="none")
            raw_loss = action_criterion(
                torch.flatten(pred_actions, end_dim=1), torch.flatten(discrete_actions, end_dim=1)
            )
            masked_loss = raw_loss * torch.flatten(attention_mask, end_dim=-1)
            action_loss = torch.sum(masked_loss) / torch.sum(attention_mask)

            return MyOutput(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=0.0 * observation_loss + 1.0 * action_loss,
            )
        else:
            return MyOutput(pred_observations=pred_observations, pred_actions=pred_actions)
