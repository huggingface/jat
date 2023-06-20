import torch
from gym.vector.vector_env import VectorEnv
from tqdm import tqdm

from gia.eval.evaluator import Evaluator
from gia.eval.rl.gia_agent import GiaAgent
from gia import GiaModel


class RLEvaluator(Evaluator):
    def _build_env(self) -> VectorEnv:  # TODO: maybe just a gym.Env ?
        raise NotImplementedError

    def _evaluate(self, model: GiaModel):
        env = self._build_env()
        gia_agent = GiaAgent(self.task, model, env.observation_space, env.action_space)

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
