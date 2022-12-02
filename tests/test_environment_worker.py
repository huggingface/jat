from functools import partial
import pytest

from gym.spaces import Dict, Box, Discrete
from gia.environment_worker import EnvironmentWorker
from gia.config.config import Config, get_config
from gia.mocks.mock_env import MockBatchedEnv, MockEnv


@pytest.mark.parametrize(
    "obs_space",
    [
        Dict(obs1=Box(-1, 1, (1, 84, 84))),
        Dict(obs1=Box(-1, 1, (1, 84, 84)), obs2=Box(-1, 1, (17,))),
        Dict(obs1=Box(-1, 1, (1, 84, 84)), obs2=Box(-1, 1, (1, 84, 84))),
        Dict(obs1=Box(-1, 1, (17,)), obs2=Box(-1, 1, (1, 84, 84))),
        Dict(obs1=Box(-1, 1, (17,))),
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        Dict(action1=Discrete(4)),
        Dict(action1=Discrete(4), action2=Box(-1, 1, (3,))),
        Dict(action1=Box(-1, 1, (3,)), action2=Box(-1, 1, (3,))),
        Dict(action1=Box(-1, 1, (3,)), action2=Discrete(4)),
        Dict(obs1=Box(-1, 1, (3,))),
    ],
)
def test_environment_worker(obs_space, action_space):
    config: Config = get_config()
    env_fn = partial(
        MockBatchedEnv,
        partial(MockEnv, obs_space, action_space),
        num_parallel=config.hyp.n_agents,
    )
    env_worker = EnvironmentWorker(config, env_fn)
    traj = env_worker.sample_trajectory()

    print(traj)
