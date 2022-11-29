from torch import nn


class MockModel:
    def __init__(self, config, obs_space, action_space) -> None:

        # assumes image observation and 1d action space
        self.config = config
        self.obs_space = obs_space
        self.action_space = action_space

        self.encoder = nn.Linear(17, 4)

    def __call__(self, obs):
        return self.forward(obs)

    def forward(self, obs):
        return self.action_space.sample()
