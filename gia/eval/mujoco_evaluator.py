import gym
import torch
from tqdm import tqdm

from gia import GiaConfig, GiaModel
from gia.config.arguments import Arguments
from gia.eval.evaluator import Evaluator
from gia.eval.mappings import DATASET_FILE_MAPPING, TASK_TO_ENV_MAPPING
from gia.eval.rl.gia_agent import GiaAgent


def make_mujoco_env(env_name, render_mode=None):
    return gym.make(env_name, render_mode=render_mode)


class MujocoEvaluator(Evaluator):
    def __init__(self, args: Arguments, task_map=TASK_TO_ENV_MAPPING, dataset_map=DATASET_FILE_MAPPING):
        self.task = "mujoco"
        self.env_names = task_map[self.task]
        self.data_filepaths = dataset_map[self.task]
        self.args = args

    def evaluate(self, model: GiaModel):
        stats = {}
        for env_name, dataset_name in zip(self.env_names, self.data_filepaths):
            stats[env_name] = self._evaluate_env(env_name, dataset_name, model)
        return stats

    @torch.no_grad()
    def _evaluate_env(self, env_name: str, dataset_name: str, model: GiaModel):
        num_envs = 1

        env = gym.vector.make(env_name, num_envs)
        gia_agent = GiaAgent(self.args, dataset_name, model, env.observation_space, env.action_space)

        returns = []
        # due to how to KV cache is used, we only can evaluate one env instance at a time
        for ep in tqdm(range(self.args.n_episodes)):
            accum_rewards = []
            done = False
            obs, info = env.reset()
            gia_agent.reset()

            while not done:
                # Compute the output of the model
                action = gia_agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                accum_rewards.append(reward[0])

            returns.append(sum(accum_rewards))
        env.close()

        return returns


if __name__ == "__main__":

    config = GiaConfig()

    args = Arguments(output_dir="tmp", n_episodes=2, task_names="mujoco-doublependulum")
    model = GiaModel(config)  # .to("cuda")

    evaluator = MujocoEvaluator(
        args,
        task_map={"mujoco": ["InvertedDoublePendulum-v4"]},
        dataset_map={"mujoco": ["mujoco-doublependulum"]},
    )
    evaluator.evaluate(model)
