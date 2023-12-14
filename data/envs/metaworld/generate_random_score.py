"""
This script generates the score for a random agent for all the metaworld environments and saves them in a dictionary.
"""

import json
import os
from multiprocessing import Pool

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np


FILENAME = "jat/eval/rl/scores_dict.json"

TASK_NAME_TO_ENV_NAME = {
    "metaworld-assembly": "assembly-v2",
    "metaworld-basketball": "basketball-v2",
    "metaworld-bin-picking": "bin-picking-v2",
    "metaworld-box-close": "box-close-v2",
    "metaworld-button-press-topdown": "button-press-topdown-v2",
    "metaworld-button-press-topdown-wall": "button-press-topdown-wall-v2",
    "metaworld-button-press": "button-press-v2",
    "metaworld-button-press-wall": "button-press-wall-v2",
    "metaworld-coffee-button": "coffee-button-v2",
    "metaworld-coffee-pull": "coffee-pull-v2",
    "metaworld-coffee-push": "coffee-push-v2",
    "metaworld-dial-turn": "dial-turn-v2",
    "metaworld-disassemble": "disassemble-v2",
    "metaworld-door-close": "door-close-v2",
    "metaworld-door-lock": "door-lock-v2",
    "metaworld-door-open": "door-open-v2",
    "metaworld-door-unlock": "door-unlock-v2",
    "metaworld-drawer-close": "drawer-close-v2",
    "metaworld-drawer-open": "drawer-open-v2",
    "metaworld-faucet-close": "faucet-close-v2",
    "metaworld-faucet-open": "faucet-open-v2",
    "metaworld-hammer": "hammer-v2",
    "metaworld-hand-insert": "hand-insert-v2",
    "metaworld-handle-press-side": "handle-press-side-v2",
    "metaworld-handle-press": "handle-press-v2",
    "metaworld-handle-pull-side": "handle-pull-side-v2",
    "metaworld-handle-pull": "handle-pull-v2",
    "metaworld-lever-pull": "lever-pull-v2",
    "metaworld-peg-insert-side": "peg-insert-side-v2",
    "metaworld-peg-unplug-side": "peg-unplug-side-v2",
    "metaworld-pick-out-of-hole": "pick-out-of-hole-v2",
    "metaworld-pick-place": "pick-place-v2",
    "metaworld-pick-place-wall": "pick-place-wall-v2",
    "metaworld-plate-slide-back-side": "plate-slide-back-side-v2",
    "metaworld-plate-slide-back": "plate-slide-back-v2",
    "metaworld-plate-slide-side": "plate-slide-side-v2",
    "metaworld-plate-slide": "plate-slide-v2",
    "metaworld-push-back": "push-back-v2",
    "metaworld-push": "push-v2",
    "metaworld-push-wall": "push-wall-v2",
    "metaworld-reach": "reach-v2",
    "metaworld-reach-wall": "reach-wall-v2",
    "metaworld-shelf-place": "shelf-place-v2",
    "metaworld-soccer": "soccer-v2",
    "metaworld-stick-pull": "stick-pull-v2",
    "metaworld-stick-push": "stick-push-v2",
    "metaworld-sweep-into": "sweep-into-v2",
    "metaworld-sweep": "sweep-v2",
    "metaworld-window-close": "window-close-v2",
    "metaworld-window-open": "window-open-v2",
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
    with Pool(32) as p:
        p.map(generate_random_score, TASK_NAME_TO_ENV_NAME.keys())
