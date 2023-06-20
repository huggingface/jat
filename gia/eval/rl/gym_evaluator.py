import gym
import torch
from gym.vector.vector_env import VectorEnv
from gia import GiaConfig, GiaModel
from gia.config.arguments import Arguments
from gia.eval.mappings import TASK_TO_ENV_MAPPING
from gia.eval.rl.rl_evaluator import RLEvaluator


class GymEvaluator(RLEvaluator):
    def _build_env(self) -> VectorEnv:
        NUM_ENVS = 1
        env_name = TASK_TO_ENV_MAPPING[self.task]
        env = gym.vector.make(env_name, NUM_ENVS)
        return env


if __name__ == "__main__":
    config = GiaConfig()

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-ant", seq_len=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GiaModel(config).to(device)

    evaluator = GymEvaluator(args, "mujoco-ant")

    evaluator.evaluate(model)
