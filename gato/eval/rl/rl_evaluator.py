import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from gato import GatoModel
from gato.eval.evaluator import Evaluator
from gato.eval.rl.gato_agent import GatoAgent
from gato.processing import GatoProcessor
from gia.eval.rl import make


class RLEvaluator(Evaluator):
    def _evaluate(self, model: GatoModel) -> float:
        def env_func():
            env = make(self.task_name)
            # env = RecordVideoV0(env, "/tmp/video", video_length=1000)
            return env

        # Due to how to KV cache is used, we only can evaluate one env instance at a time
        vec_env = gym.vector.SyncVectorEnv(env_fns=[env_func])
        processor = GatoProcessor()  # Ideally, model.config
        gia_agent = GatoAgent(model, processor, self.task_name, num_envs=1)

        # Initialize the environment and the agent
        obs, info = vec_env.reset()
        gia_agent.reset()
        returns = []
        episode_reward = 0

        progress_bar = tqdm(total=self.args.n_episodes)
        while len(returns) < self.args.n_episodes:
            progress_bar.update(1)
            with torch.inference_mode():
                action = gia_agent.get_action(obs)
            obs, reward, truncated, terminated, info = vec_env.step(action)
            episode_reward += reward[0]
            if terminated or truncated:
                gia_agent.reset()
                returns.append(episode_reward)
                episode_reward = 0
        vec_env.close()

        return np.mean(returns)  # TODO: add std for more detailed logging
