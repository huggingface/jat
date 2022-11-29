import numpy as np
import pytest
import torch
from gym.spaces import Box, Dict, Discrete

from gia.config.config import get_config
from gia.model.encoder import default_make_encoder_func
from gia.model.actor_critic import create_actor_critic


@pytest.mark.parametrize(
    "obs_space",
    [
        Dict(
            {
                "obs_1d": Box(-1, 1, shape=(21,)),
            }
        ),
        Dict(
            {
                "obs_1d": Box(-1, 1, shape=(21,)),
                "obs_3d": Box(-1, 1, shape=(3, 84, 84)),
            }
        ),
        Dict(
            {
                "obs_1d": Box(-1, 1, shape=(21,)),
                "obs_3d": Box(-1, 1, shape=(3, 84, 84)),
                "obs_3d_2": Box(-1, 1, shape=(3, 64, 64)),
            }
        ),
        Dict(
            {
                "obs": Box(-1, 1, shape=(21,)),
            }
        ),
        Dict(
            {
                "obs": Box(-1, 1, shape=(3, 84, 84)),
            }
        ),
        Dict(
            {
                "obs": Box(-1, 1, shape=(3, 84, 84)),
                "measurements": Box(-1, 1, shape=(21,)),
            }
        ),
    ],
)
def test_default_make_encoder_func(obs_space):
    config = get_config()
    encoder = default_make_encoder_func(config, obs_space)
    obs = obs_space.sample()
    for k in obs.keys():
        obs[k] = torch.from_numpy(np.expand_dims(obs[k], 0))

    output = encoder(obs)

    assert set(encoder.obs_keys) == set(obs_space.keys())
    assert output.shape == (1, 512 * len(obs_space))


@pytest.mark.parametrize(
    "obs_space",
    [
        Dict(
            {
                "obs_1d": Box(-1, 1, shape=(21,)),
                "obs_3d": Box(-1, 1, shape=(3, 84, 84)),
                "obs_3d_2": Box(-1, 1, shape=(3, 64, 64)),
            }
        )
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        Dict(
            {
                "discrete": Discrete(5),
                "continuous": Box(-1, 1, shape=(7,)),
            }
        )
    ],
)
def test_model_builder(obs_space, action_space):
    config = get_config()
    actor_critic = create_actor_critic(config, obs_space, action_space)
