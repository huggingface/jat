from abc import ABC

import torch
from torch import nn

from gia.model.model_utils import ModelModule
from gia.config.config import Config


class ModelCore(ModelModule, ABC):
    def __init__(self, config: Config):
        super().__init__(config)
        self.core_output_size = -1  # to be overridden in derived classes

    def get_out_size(self) -> int:
        return self.core_output_size


class ModelCoreRNN(ModelCore):
    def __init__(self, config, input_size):
        super().__init__(config)

        self.config = config
        self.is_gru = False

        if config.rnn_type == "gru":
            self.core = nn.GRU(input_size, config.rnn_size, config.rnn_num_layers)
            self.is_gru = True
        elif config.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, config.rnn_size, config.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {config.rnn_type}")

        self.core_output_size = config.rnn_size
        self.rnn_num_layers = config.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.config.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.config.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class ModelCoreIdentity(ModelCore):
    """A noop core (no recurrency)."""

    def __init__(self, config, input_size):
        super().__init__(config)
        self.config = config
        self.core_output_size = input_size

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states):
        return head_output, fake_rnn_states


def default_make_core_func(config: Config, core_input_size: int) -> ModelCore:
    if config.use_rnn:
        core = ModelCoreRNN(config, core_input_size)
    else:
        core = ModelCoreIdentity(config, core_input_size)

    return core
