import torch

from gia.mocks.mock_config import MockConfig
from gia.mocks.mock_env import MockBatchedEnv, MockImageEnv
from gia.mocks.mock_model import MockModel
from gia.replay_buffer import ReplayBuffer


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

        self._update_model_weights()

        return buffer

    def get_env_info(self):
        return self.env.observation_space, self.env.action_space

    def _update_model_weights(self):
        pass
