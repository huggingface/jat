import gym
from gym.vector.vector_env import VectorEnv

from gia.eval.mappings import TASK_TO_ENV_MAPPING
from gia.eval.rl.rl_evaluator import RLEvaluator


class GymEvaluator(RLEvaluator):
    def _build_env(self) -> VectorEnv:
        NUM_ENVS = 1
        env_name = TASK_TO_ENV_MAPPING[self.task]
        env = gym.vector.make(env_name, NUM_ENVS)
        return env
