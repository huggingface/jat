import torch
from gym.spaces import Box, Dict, Discrete

from gia.utils.tensor_dict import TensorDict
from gia.utils.utils import check_space_is_flat_dict, to_torch_dtype
from gia.config.config import Config


class ReplayBuffer:
    def __init__(self, config: Config, obs_space: Dict, action_space: Dict):
        self.obs_space = obs_space
        self.action_space = action_space
        self.hyp = config.hyp

        self.buffer = TensorDict(
            terms=torch.ones(self.hyp.rollout_length, self.hyp.n_agents, dtype=torch.bool),
            truncs=torch.ones(self.hyp.rollout_length, self.hyp.n_agents, dtype=torch.bool),
            rewards=torch.ones(self.hyp.rollout_length, self.hyp.n_agents, dtype=torch.float32),
            values=torch.ones(self.hyp.rollout_length, self.hyp.n_agents, dtype=torch.float32),
        )
        self.buffer["observations"] = self._build_buffer_from_space(obs_space)
        self.buffer["actions"] = self._build_buffer_from_space(action_space)

    def to(self, device):
        # Moves buffer to device
        pass

    def __getitem__(self, key):
        return self.buffer[key]

    def _build_buffer_from_space(self, space: Dict):
        check_space_is_flat_dict(space)
        sub_buffer = TensorDict()
        for k, v in space.items():
            if isinstance(v, Box):
                sub_buffer[k] = torch.ones(
                    (self.hyp.rollout_length, self.hyp.n_agents, *v.shape), dtype=to_torch_dtype(v.dtype)
                )
                # TODO split obs and actions
                sub_buffer[f"{k}_old_logits"] = torch.ones(
                    self.hyp.rollout_length, self.hyp.n_agents, *v.shape, dtype=torch.float32
                )
            elif isinstance(v, Discrete):
                sub_buffer[k] = torch.ones((self.hyp.rollout_length, self.hyp.n_agents, 1), dtype=torch.int32)
                sub_buffer[f"{k}_old_logits"] = torch.ones(
                    self.hyp.rollout_length, self.hyp.n_agents, 1, dtype=torch.float32
                )

        return sub_buffer
