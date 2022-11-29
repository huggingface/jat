from typing import List, Optional

import torch
from gym import spaces
from gym.spaces import Box, Dict
from torch import Tensor, nn

from gia.model.model_utils import ModelModule, calc_num_elements, create_mlp, model_device, nonlinearity
from gia.config.config import Config

# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Encoder(ModelModule):
    def __init__(self, config: Config):
        super().__init__(config)

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


class MultiInputEncoder(Encoder):
    def __init__(self, config: Config, obs_space: Dict):
        super().__init__(config)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        out_size = 0

        for obs_key in self.obs_keys:
            shape = obs_space[obs_key].shape

            if len(shape) == 1:
                encoder_fn = MlpEncoder
            elif len(shape) > 1:
                encoder_fn = make_img_encoder
            else:
                raise NotImplementedError(f"Unsupported observation space {obs_space}")

            self.encoders[obs_key] = encoder_fn(config, obs_space[obs_key])
            out_size += self.encoders[obs_key].get_out_size()

        self.encoder_out_size = out_size

    def forward(self, obs_dict):
        if len(self.obs_keys) == 1:
            key = self.obs_keys[0]
            return self.encoders[key](obs_dict[key])

        encodings = []
        for key in self.obs_keys:
            x = self.encoders[key](obs_dict[key])
            encodings.append(x)

        return torch.cat(encodings, 1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class MlpEncoder(Encoder):
    def __init__(self, config: Config, obs_space: Box):
        super().__init__(config)

        mlp_layers: List[int] = config.model.encoder_mlp_layers
        self.mlp_head = create_mlp(mlp_layers, obs_space.shape[0], nonlinearity(config))
        if len(mlp_layers) > 0:
            self.mlp_head = torch.jit.script(self.mlp_head)
        self.encoder_out_size = calc_num_elements(self.mlp_head, obs_space.shape)

    def forward(self, obs: Tensor):
        x = self.mlp_head(obs)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class ConvEncoderImpl(nn.Module):
    """
    After we parse all the configuration and figure out the exact architecture of the model,
    we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
    fusion).
    """

    def __init__(self, obs_shape, conv_filters: List, extra_mlp_layers: List[int], activation: nn.Module):
        super().__init__()

        conv_layers = []
        for layer in conv_filters:
            if layer == "maxpool_2x2":
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(activation)
            else:
                raise NotImplementedError(f"Layer {layer} not supported!")

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)
        self.mlp_layers = create_mlp(extra_mlp_layers, self.conv_head_out_size, activation)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x


class ConvEncoder(Encoder):
    def __init__(self, config: Config, obs_space: Box):
        super().__init__(config)

        input_channels = obs_space.shape[0]
        if config.model.encoder_conv_architecture == "convnet_simple":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif config.model.encoder_conv_architecture == "convnet_impala":
            conv_filters = [[input_channels, 16, 8, 4], [16, 32, 4, 2]]
        elif config.model.encoder_conv_architecture == "convnet_atari":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 64, 3, 1]]
        else:
            raise NotImplementedError(f"Unknown encoder architecture {config.model.encoder_conv_architecture}")

        activation = nonlinearity(self.config)
        extra_mlp_layers: List[int] = config.model.encoder_conv_mlp_layers
        enc = ConvEncoderImpl(obs_space.shape, conv_filters, extra_mlp_layers, activation)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_space.shape)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs: Tensor) -> Tensor:
        return self.enc(obs)


class ResBlock(nn.Module):
    def __init__(self, config: Config, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(config),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(config),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    def __init__(self, config: Config, obs_space):
        super().__init__(config)

        input_ch = obs_space.shape[0]

        if config.encoder_conv_architecture == "resnet_impala":
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {config.encode_conv_architecture}")

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock(config, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = nonlinearity(config)
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        self.mlp_layers = create_mlp(config.encoder_conv_mlp_layers, self.conv_head_out_size, activation)

        # should we do torch.jit here?

        self.encoder_out_size = calc_num_elements(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_img_encoder(config: Config, obs_space: Box) -> Encoder:
    """Make (most likely convolutional) encoder for image-based observations."""
    if config.model.encoder_conv_architecture.startswith("convnet"):
        return ConvEncoder(config, obs_space)
    elif config.model.encoder_conv_architecture.startswith("resnet"):
        return ResnetEncoder(config, obs_space)
    else:
        raise NotImplementedError(f"Unknown convolutional architecture {config.model.encoder_conv_architecture}")


def default_make_encoder_func(config: Config, obs_space: Dict) -> Encoder:
    """
    Analyze the observation space and create either a convolutional or an MLP encoder depending on
    whether this is an image-based environment or environment with vector observations.
    """
    # we only support dict observation spaces - envs with non-dict obs spaces use a wrapper
    # main subspace used to determine the encoder type is called "obs". For envs with multiple subspaces,
    # this function needs to be overridden (see vizdoom or dmlab encoders for example)
    return MultiInputEncoder(config, obs_space)
