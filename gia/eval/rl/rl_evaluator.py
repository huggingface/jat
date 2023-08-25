import numpy as np
from tqdm import tqdm

from gia import GiaModel
from gia.eval.evaluator import Evaluator
from gia.eval.rl import make
from gia.eval.rl.gia_agent import GiaAgent
from gia.processing import GiaProcessor


class RLEvaluator(Evaluator):
    def _evaluate(self, model: GiaModel) -> float:
        env = make(self.task_name)
        processor = GiaProcessor()  # Ideally, model.config
        gia_agent = GiaAgent(model, processor, self.task_name, num_envs=1)

        returns = []
        # due to how to KV cache is used, we only can evaluate one env instance at a time
        for ep in tqdm(range(self.args.n_episodes)):
            accum_rewards = []
            done = False
            obs, info = env.reset()
            gia_agent.reset()

            while not done:
                # Compute the output of the model
                action = gia_agent.get_action([obs])[0]
                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                accum_rewards.append(reward)

            returns.append(sum(accum_rewards))
        env.close()

        return np.mean(returns)  # TODO: add std for more detailed logging
