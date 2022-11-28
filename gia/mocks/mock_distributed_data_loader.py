import random

from .mock_constants import MOCK_OBS_SPACE, MOCK_ACTION_SPACE
from gia.replay_buffer import ReplayBuffer


class MockDistributedDataLoader:
    def __init__(
        self,
        obs_space=MOCK_OBS_SPACE,
        action_space=MOCK_ACTION_SPACE,
        n_agents=8,
        rollout_length=16,
        n_epochs=4,
        mini_batch_size=4,
    ):
        assert mini_batch_size <= n_agents
        assert n_agents % mini_batch_size == 0

        self.iters_per_epoch = n_agents // mini_batch_size
        self.obs_space = obs_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.rollout_length = rollout_length
        self.n_epochs = n_epochs
        self.minibatch_size = mini_batch_size

        self._double_buffer = [
            ReplayBuffer(obs_space, action_space, n_agents, rollout_length),
            ReplayBuffer(obs_space, action_space, n_agents, rollout_length),
        ]
        self._current_read_buffer = 0
        self._current_write_buffer = 1

    def __iter__(self):
        for epoch in range(self.n_epochs):
            indices = list(range(self.n_agents))
            random.shuffle(indices)

            for i in range(self.iters_per_epoch):
                buf = self._double_buffer[self._current_read_buffer]
                sub_indices = indices[i * self.minibatch_size : (i + 1) * self.minibatch_size]
                mini_batch = buf[sub_indices]
                yield mini_batch

    def __len__(self):
        return 8
