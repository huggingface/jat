import pytest
import torch
import numpy as np
from gym.spaces import Dict, Discrete, Box

from gia.replay_buffer import ReplayBuffer
from gia.utils.utils import to_torch_dtype


@pytest.mark.parametrize(
    "obs_space", [Dict(image=Box(0, 1, shape=(3, 32, 40), dtype=np.float32), discrete=Discrete(4))]
)
@pytest.mark.parametrize("action_space", [Dict(cont=Box(0, 1, shape=(17,), dtype=np.float32), discrete=Discrete(3))])
def test_replay_buffer(obs_space, action_space):
    buffer = ReplayBuffer(obs_space, action_space, 7, 11)

    subset = buffer[:2]

    assert subset["dones"].shape == (2, 7)
    assert subset["dones"].dtype == torch.bool
    assert subset["rewards"].shape == (2, 7)
    assert subset["rewards"].dtype == torch.float32
    assert subset["values"].shape == (2, 7)
    assert subset["values"].dtype == torch.float32

    for k, v in obs_space.items():
        if isinstance(v, Box):
            assert subset["observations"][k].shape == (2, 7, *v.shape)
            assert subset["observations"][k].dtype == to_torch_dtype(v.dtype)
        elif isinstance(v, Discrete):
            assert subset["observations"][k].shape == (2, 7, 1)
            assert subset["observations"][k].dtype == torch.int32
        else:
            assert 0, "invalid space in space"

    for k, v in action_space.items():
        if isinstance(v, Box):
            assert subset["actions"][k].shape == (2, 7, *v.shape)
            assert subset["actions"][k].dtype == to_torch_dtype(v.dtype)
            assert subset["actions"][f"{k}_old_logits"].shape == (2, 7, *v.shape)
            assert subset["actions"][f"{k}_old_logits"].dtype == torch.float32
        elif isinstance(v, Discrete):
            assert subset["actions"][k].shape == (2, 7, 1)
            assert subset["actions"][k].dtype == torch.int32
            assert subset["actions"][f"{k}_old_logits"].shape == (2, 7, 1)
            assert subset["actions"][f"{k}_old_logits"].dtype == torch.float32
        else:
            assert 0, "invalid space in space"


if __name__ == "__main__":
    obs_space = Dict(image=Box(0, 1, shape=(3, 32, 40), dtype=np.float32), discrete=Discrete(4))
    action_space = Dict(cont=Box(0, 1, shape=(17,), dtype=np.float32), discrete=Discrete(3))

    test_replay_buffer(obs_space, action_space)
