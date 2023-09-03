import json

import gymnasium as gym
import metaworld
import numpy as np
import torch
from gymnasium.experimental.wrappers import RecordVideoV0

from gia import GiaModel
from gia.eval.rl import make
from gia.eval.rl.gia_agent import GiaAgent
from gia.processing import GiaProcessor


def main():
    with open("gia/eval/rl/scores_dict.json", "r") as file:
        scores_dict = json.load(file)

    mean_expert_score = scores_dict[task_name]["expert"]["mean"]
    mean_random_score = scores_dict[task_name]["random"]["mean"]

    def env_func():
        env = make(task_name, render_mode="rgb_array")
        # env = RecordVideoV0(env, f"/tmp/video_{task_name}", video_length=500)
        return env

    vec_env = gym.vector.SyncVectorEnv(env_fns=[env_func])

    # Build the model
    model = GiaModel.from_pretrained("checkpoints/739004/checkpoint-47500")  # .to("cuda")
    processor = GiaProcessor()
    agent = GiaAgent(model, processor, task_name, deterministic=False)

    # Initialize the environment and the agent
    obs, info = vec_env.reset()
    agent.reset()

    # Run the agent for 10_000 steps
    all_episode_rewards = []
    episode_reward = 0
    for t in range(500):
        with torch.inference_mode():
            action = agent.get_action(obs)
        obs, reward, truncated, terminated, info = vec_env.step(action)
        episode_reward += reward
        if terminated or truncated:
            obs, info = vec_env.reset()
            agent.reset()
            all_episode_rewards.append(episode_reward)
            episode_reward = 0
    vec_env.close()

    normalized_score = (np.mean(all_episode_rewards) - mean_random_score) / (mean_expert_score - mean_random_score)

    print(normalized_score)


task_names = [
    "mujoco-ant",
    "mujoco-doublependulum",
    "mujoco-halfcheetah",
    "mujoco-hopper",
    "mujoco-humanoid",
    "mujoco-pendulum",
    "mujoco-pusher",
    "mujoco-reacher",
    "mujoco-standup",
    "mujoco-swimmer",
    "mujoco-walker",
]

for task_name in task_names:
    main()

