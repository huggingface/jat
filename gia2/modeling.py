from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor, nn
from transformers import GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput

from gia2.config import Gia2Config
from gia2.utils import compute_ce_loss, compute_mse_loss, cyclic_expand_dim


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.in1 = nn.InstanceNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.in2 = nn.InstanceNorm2d(out_channels)
#         self.dropout = nn.Dropout(0.5)
#         self.leakyrelu = nn.LeakyReLU(0.2)

#         # For the skip connection
#         self.skip = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.InstanceNorm2d(out_channels)
#             )

#     def forward(self, x):
#         residual = self.skip(x)
#         x = self.leakyrelu(self.in1(self.conv1(x)))
#         x = self.dropout(x)
#         x = self.in2(self.conv2(x))
#         x += residual
#         x = self.leakyrelu(x)
#         return x


# class ImageEncoder2(nn.Module):
#     def __init__(self, hidden_size: int):
#         super().__init__()

#         # Encoder with Residual Blocks
#         self.enc1 = ResidualBlock(4, 16, stride=2)
#         self.enc2 = ResidualBlock(16, 32, stride=2)
#         self.enc3 = ResidualBlock(32, 64, stride=2)
#         self.enc4 = ResidualBlock(64, 128, stride=2)

#         # Bottleneck fully connected layers
#         self.fc = nn.Linear(128 * 6 * 6, hidden_size)

#     def forward(self, x):
#         # Encoder
#         x = self.enc1(x)  # 4 x 84 x 84 -> 16 x 42 x 42
#         x = self.enc2(x)  # 16 x 42 x 42 -> 32 x 21 x 21
#         x = self.enc3(x)  # 32 x 21 x 21 -> 64 x 11 x 11
#         x = self.enc4(x)  # 64 x 10 x 10 -> 128 x 6 x 6

#         # Flatten and pass through bottleneck
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# class ImageDecoder2(nn.Module):
#     def __init__(self, hidden_size: int):
#         super().__init__()
#         self.fc2 = nn.Linear(hidden_size, 128 * 6 * 6)

#         # Decoder with Residual Blocks
#         self.dec1 = ResidualBlock(128, 64, stride=2)
#         self.dec2 = ResidualBlock(64, 32, stride=2)
#         self.dec3 = ResidualBlock(32, 16, stride=2)
#         self.dec4 = ResidualBlock(16, 4, stride=2)

#     def forward(self, x):
#         x = self.fc2(x)
#         x = x.view(x.size(0), 128, 6, 6)

#         # Decoder
#         x = self.dec1(x)  # 128 x 6 x 6 -> 64 x 12 x 12
#         x = self.dec2(x)  # 64 x 12 x 12 -> 32 x 24 x 24
#         x = self.dec3(x)  # 32 x 24 x 24 -> 16 x 48 x 48
#         x = torch.sigmoid(self.dec4(x))
#         return x


# class ResBlock(nn.Module):
#     """
#     Residual block with instance normalization.

#     Args:
#         num_channels (`int`):
#             Number of input and output channels.
#     """

#     def __init__(self, shape: int) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(shape[0], shape[0], kernel_size=3, padding=1)
#         self.in1 = nn.LayerNorm(shape)
#         self.conv2 = nn.Conv2d(shape[0], shape[0], kernel_size=3, padding=1)
#         self.in2 = nn.LayerNorm(shape)

#     def forward(self, x: FloatTensor) -> FloatTensor:
#         residual = x
#         out = F.leaky_relu(self.in1(self.conv1(x)))
#         out = self.in2(self.conv2(out))
#         out += residual
#         return F.leaky_relu(out)


# class ImageEncoder(nn.Module):
#     """
#     Image encoder. Input is a batch of images of shape (N, 4, 84, 84).

#     Args:
#         hidden_size (`int`):
#             Size of the output hidden state.
#     """

#     def __init__(self, hidden_size: int) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
#         self.res1 = ResBlock((32, 42, 42))
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.res2 = ResBlock((64, 21, 21))
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.res3 = ResBlock((128, 10, 10))
#         self.fc = nn.Linear(128 * 10 * 10, hidden_size)

#     def forward(self, x: FloatTensor) -> FloatTensor:
#         x = F.max_pool2d(F.leaky_relu(self.conv1(x)), (2, 2))  # (4 x 84 x 84) -> (32 x 42 x 42)
#         x = self.res1(x)
#         x = F.max_pool2d(F.leaky_relu(self.conv2(x)), (2, 2))  # (32 x 42 x 42) -> (64 x 21 x 21)
#         x = self.res2(x)
#         x = F.max_pool2d(F.leaky_relu(self.conv3(x)), (2, 2))  # (64 x 21 x 21) -> (128 x 10 x 10)
#         x = self.res3(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#         return x


# class ImageDecoder(nn.Module):
#     """
#     Image decoder. Output is a batch of images of shape (N, 4, 84, 84).

#     Args:
#         hidden_size (`int`):
#             Size of the input hidden state.
#     """

#     def __init__(self, hidden_size: int) -> None:
#         super().__init__()
#         self.fc = nn.Linear(hidden_size, 128 * 10 * 10)
#         self.convt1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
#         self.res1 = ResBlock((64, 21, 21))
#         self.convt2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
#         self.res2 = ResBlock((32, 42, 42))
#         self.convt3 = nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1)
#         self.res3 = ResBlock((4, 84, 84))

#     def forward(self, x: FloatTensor) -> FloatTensor:
#         x = self.fc(x)
#         x = x.view(x.size(0), 128, 10, 10)  # (128 * 10 * 10) -> (128 x 10 x 10)
#         x = F.leaky_relu(self.convt1(x))  # (128 x 10 x 10) -> (64 x 21 x 21)
#         x = self.res1(x)
#         x = F.leaky_relu(self.convt2(x))  # (64 x 21 x 21) -> (32 x 42 x 42)
#         x = self.res2(x)
#         x = F.leaky_relu(self.convt3(x))  # (32 x 42 x 42) -> (4 x 84 x 84)
#         x = F.sigmoid(self.res3(x))
#         return x

from gia2.resnet import ImageDecoder, ImageEncoder


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
        self.discrete_decoder = nn.Linear(config.hidden_size, 18, bias=False)
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
            inputs_embeds_observations = self.image_encoder(image_observations)

        if continuous_actions is not None:
            batch_size, seq_len, action_size = continuous_actions.shape
            continuous_actions = cyclic_expand_dim(continuous_actions, self.config.max_continuous_size)
            inputs_embeds_actions = self.continuous_encoder(continuous_actions)
        elif discrete_actions is not None:
            batch_size, seq_len = discrete_actions.shape
            inputs_embeds_actions = self.discrete_encoder(discrete_actions)

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
            pred_observations = self.image_decoder(hidden_observations)
            if return_loss:
                observation_loss = compute_mse_loss(
                    pred_observations[:, :-1], image_observations[:, 1:], attention_mask[:, 1:], weights=loss_weight
                )

        if continuous_actions is not None:
            pred_actions = self.continuous_decoder(hidden_actions)
            if return_loss:
                action_loss = compute_mse_loss(pred_actions, continuous_actions, attention_mask, weights=loss_weight)
            pred_actions = pred_actions[..., :action_size]
        elif discrete_actions is not None:
            pred_actions = self.discrete_decoder(hidden_actions)
            if return_loss:
                action_loss = compute_ce_loss(pred_actions, discrete_actions, attention_mask, weights=loss_weight)

        if return_loss:
            return GIA2Output(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
                observation_loss=observation_loss,
                action_loss=action_loss,
                loss=1.0 * observation_loss + 0.0 * action_loss,
            )
        else:
            return GIA2Output(
                pred_observations=pred_observations,
                pred_actions=pred_actions,
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
