import torch
from gym.spaces import Dict, Box, Discrete
from gia.utils.utils import check_space_is_flat_dict, to_torch_dtype
from gia.utils.tensor_dict import TensorDict

class ReplayBuffer():
    def __init__(self, obs_space, action_space, n_agents, rollout_length):
        self.obs_space = obs_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.rollout_length = rollout_length
        
        self.buffer = TensorDict(
            dones=torch.ones(n_agents, rollout_length, dtype=torch.bool),
            rewards=torch.ones(n_agents, rollout_length, dtype=torch.float32),
            values=torch.ones(n_agents, rollout_length, dtype=torch.float32)
        )

        self.buffer["obs"] = self._build_buffer_from_space(obs_space)
        self.buffer["action"] = self._build_buffer_from_space(action_space)

    def to(self, device):
        # Moves buffer to device
        pass 

    def __getitem__(self, key):
        return self.buffer[key]

    def _build_buffer_from_space(self, space : Dict):
        check_space_is_flat_dict(space)
        sub_buffer = {}
        for k,v in space.items():
            if isinstance(v, Box):
                sub_buffer[k] = torch.ones((self.n_agents, self.rollout_length, *v.shape), dtype=to_torch_dtype(v.dtype))
                sub_buffer[f"{k}_old_logits"] = torch.ones(self.n_agents, self.rollout_length, *v.shape, dtype=torch.float32)
            elif isinstance(v, Discrete):
                sub_buffer[k] = torch.ones((self.n_agents, self.rollout_length, 1), dtype=torch.int32)
                sub_buffer[f"{k}_old_logits"] = torch.ones(self.n_agents, self.rollout_length, 1,  dtype=torch.float32)

        return sub_buffer

        

