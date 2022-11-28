import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

from gia.mocks.mock_config import MockConfig
from gia.mocks.mock_env import MockBatchedEnv, MockImageEnv
from gia.mocks.mock_model import MockModel
from gia.replay_buffer import ReplayBuffer

from gia.utils.utils import _call_remote_method


WORKER_NAME = "ENV_WORKER_{}"


class EnvironmentWorker:
    def __init__(self, config=None, model=None):

        if config is None:
            self.config = MockConfig()
        else:
            self.config = config

        # use an env registry
        self.env = MockBatchedEnv(MockImageEnv, num_parallel=self.config.n_parallel_agents)

        if model is None:
            self.model = MockModel(config, self.env.observation_space, self.env.action_space)
        else:
            self.model = model
        self._prev_obs = self.env.reset()

    def sample_trajectory(self) -> ReplayBuffer:
        buffer = ReplayBuffer(
            self.env.observation_space,
            self.env.action_space,
            n_agents=self.config.n_parallel_agents,
            rollout_length=self.config.rollout_length,
        )

        for i in range(self.config.rollout_length):
            _action = self.model(self._prev_obs)
            action = self.env.sample_actions()
            next_obs, reward, done, info = self.env.step(action)

            for k, v in action.items():
                buffer["actions"][k][i] = torch.from_numpy(v)

            for k, v in self._prev_obs.items():
                buffer["observations"][k][i] = torch.from_numpy(v)

            buffer["rewards"][i] = torch.from_numpy(reward)
            buffer["dones"][i] = torch.from_numpy(done)

            self._prev_obs = next_obs

        self.update_model_weights()

        return buffer

    def get_env_info(self):
        return self.env.observation_space, self.env.action_space

    def update_model_weights(self):
        print("updating weight not implemented in ", self.__class__)


class DistributedEnvironmentWorker(EnvironmentWorker):
    def __init__(self, model_server_rref: RRef, config=None, model=None):
        super().__init__(config, model)
        self.id = rpc.get_worker_info().id
        self.model_server_rref = model_server_rref

    def update_model_weights(self):
        from gia.learner_worker import LearnerWorker  # here because of circular import

        print("requesting updated model weights")
        try:  # this can throw an exception when training ends
            weights = rpc_sync(
                self.model_server_rref.owner(),
                _call_remote_method,
                args=(LearnerWorker.get_latest_model_weights, self.model_server_rref),
            )
            print(weights)

        except RuntimeError as e:
            print("RuntimeError", e, type(e))
