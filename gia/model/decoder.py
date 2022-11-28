from abc import ABC
from typing import List

import torch

from gia.model.model_utils import calc_num_elements
from gia.model.model_utils import ModelModule, create_mlp, nonlinearity
from gia.config.config import Config


class Decoder(ModelModule, ABC):
    pass


class MlpDecoder(Decoder):
    def __init__(self, config: Config, decoder_input_size: int):
        super().__init__(config)
        self.core_input_size = decoder_input_size
        decoder_layers: List[int] = config.decoder_mlp_layers
        activation = nonlinearity(config)
        self.mlp = create_mlp(decoder_layers, decoder_input_size, activation)
        if len(decoder_layers) > 0:
            self.mlp = torch.jit.script(self.mlp)

        self.decoder_out_size = calc_num_elements(self.mlp, (decoder_input_size,))

    def forward(self, core_output):
        return self.mlp(core_output)

    def get_out_size(self):
        return self.decoder_out_size


def default_make_decoder_func(config: Config, core_input_size: int) -> Decoder:
    return MlpDecoder(config, core_input_size)
