"""
This script generates the score for a random agent for all the metaworld environments and saves them in a dictionary.
"""

import json
import os
from multiprocessing import Pool

import gymnasium as gym
import numpy as np


FILENAME = "jat/eval/rl/scores_dict.json"

TASK_NAME_TO_ENV_NAME = {
    "mujoco-ant": "Ant-v4",
    "mujoco-doublependulum": "InvertedDoublePendulum-v4",
    "mujoco-halfcheetah": "HalfCheetah-v4",
    "mujoco-hopper": "Hopper-v4",
    "mujoco-humanoid": "Humanoid-v4",
    "mujoco-pendulum": "InvertedPendulum-v4",
    "mujoco-pusher": "Pusher-v4",
    "mujoco-reacher": "Reacher-v4",
    "mujoco-standup": "HumanoidStandup-v4",
    "mujoco-swimmer": "Swimmer-v4",
    "mujoco-walker": "Walker2d-v4",
}

TOT_NUM_TIMESTEPS = 1_000_000


def generate_random_score(task_name):
    # Make the environment
    env_name = TASK_NAME_TO_ENV_NAME[task_name]
    env = gym.make(env_name)
    env.reset()

    # Initialize the variables
    all_episode_rewards = []
    tot_episode_rewards = 0  # for one episode
    num_timesteps = 0
    terminated = truncated = False
    while num_timesteps < TOT_NUM_TIMESTEPS or not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        tot_episode_rewards += reward
        num_timesteps += 1
        if terminated or truncated:
            env.reset()
            all_episode_rewards.append(tot_episode_rewards)
            tot_episode_rewards = 0

    # Load the scores dictionary
    if not os.path.exists(FILENAME):
        scores_dict = {}
    else:
        with open(FILENAME, "r") as file:
            scores_dict = json.load(file)

    # Add the random scores to the dictionary
    if task_name not in scores_dict:
        scores_dict[task_name] = {}
    scores_dict[task_name]["random"] = {"mean": np.mean(all_episode_rewards), "std": np.std(all_episode_rewards)}

    # Save the dictionary to a file
    with open(FILENAME, "w") as file:
        scores_dict = {
            task: {agent: scores_dict[task][agent] for agent in sorted(scores_dict[task])}
            for task in sorted(scores_dict)
        }
        json.dump(scores_dict, file, indent=4)


if __name__ == "__main__":
    with Pool(11) as p:
        p.map(generate_random_score, TASK_NAME_TO_ENV_NAME.keys())
