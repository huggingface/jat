from dataclasses import dataclass
from functools import partial
import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_sync

from gym import spaces
from gia.envs.atari import make_atari_env
from gia.envs.wrappers import autowrap

from gia.model.actor_critic import ActorCritic
from gia.replay_buffer import ReplayBuffer
from gia.utils.action_distributions import to_action_space
from gia.config.globals import global_context
from gia.config.config import Config
from gia.utils.utils import _call_remote_method

WORKER_NAME = "ENV_WORKER_{}"


@dataclass
class EnvInfo:
    observation_space: spaces.Dict
    action_space: spaces.Dict


class EnvironmentWorker:
    def __init__(self, config: Config, env_func):
        self.config = config
        # TODO: use an env registry
        self.env = env_func()
        self.env = autowrap(self.env)
        self.model: ActorCritic = global_context().model_factory.make_actor_critic_func(
            config, self.env.observation_space, self.env.action_space
        )
        self._prev_obs, _ = self.env.reset()

    def sample_trajectory(self) -> ReplayBuffer:
        buffer = ReplayBuffer(
            self.config,
            self.env.observation_space,
            self.env.action_space,
        )

        for i in range(self.config.hyp.rollout_length):
            prev_obs = {}  # TODO: wrap env so all envs produce torch tensors
            for k, v in self._prev_obs.items():
                prev_obs[k] = torch.from_numpy(v)
            normalized_obs = self.model.normalize_obs(prev_obs)
            result = self.model(normalized_obs, None)
            action = to_action_space(result["actions"], self.env.action_space)

            # action = self.env.sample_actions()

            next_obs, reward, term, trunc, info = self.env.step(action)

            for k, v in action.items():
                buffer["actions"][k][i] = torch.from_numpy(v)

            for k, v in self._prev_obs.items():
                buffer["observations"][k][i] = torch.from_numpy(v)

            buffer["rewards"][i] = torch.from_numpy(reward)
            buffer["terms"][i] = torch.from_numpy(term)
            buffer["truncs"][i] = torch.from_numpy(trunc)

            self._prev_obs = next_obs

        self.update_model_weights()

        return buffer

    def get_env_info(self):
        return EnvInfo(self.env.observation_space, self.env.action_space)

    def update_model_weights(self):
        print("updating weight not implemented in ", self.__class__)


class DistributedEnvironmentWorker(EnvironmentWorker):
    def __init__(self, model_server_rref: RRef, config: Config):
        super().__init__(config)
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


if __name__ == "__main__":

    config = Config.build()
    env_worker = EnvironmentWorker(config, partial(make_atari_env, "atari_pong", config, None))
    traj = env_worker.sample_trajectory()
