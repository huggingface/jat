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
    "babyai-action-obj-door": "BabyAI-ActionObjDoor-v0",
    "babyai-blocked-unlock-pickup": "BabyAI-BlockedUnlockPickup-v0",
    "babyai-boss-level-no-unlock": "BabyAI-BossLevelNoUnlock-v0",
    "babyai-boss-level": "BabyAI-BossLevel-v0",
    "babyai-find-obj-s5": "BabyAI-FindObjS5-v0",
    "babyai-go-to-door": "BabyAI-GoToDoor-v0",
    "babyai-go-to-imp-unlock": "BabyAI-GoToImpUnlock-v0",
    "babyai-go-to-local": "BabyAI-GoToLocal-v0",
    "babyai-go-to-obj-door": "BabyAI-GoToObjDoor-v0",
    "babyai-go-to-obj": "BabyAI-GoToObj-v0",
    "babyai-go-to-red-ball-grey": "BabyAI-GoToRedBallGrey-v0",
    "babyai-go-to-red-ball-no-dists": "BabyAI-GoToRedBallNoDists-v0",
    "babyai-go-to-red-ball": "BabyAI-GoToRedBall-v0",
    "babyai-go-to-red-blue-ball": "BabyAI-GoToRedBlueBall-v0",
    "babyai-go-to-seq": "BabyAI-GoToSeq-v0",
    "babyai-go-to": "BabyAI-GoTo-v0",
    "babyai-key-corridor": "BabyAI-KeyCorridor-v0",
    "babyai-mini-boss-level": "BabyAI-MiniBossLevel-v0",
    "babyai-move-two-across-s8n9": "BabyAI-MoveTwoAcrossS8N9-v0",
    "babyai-one-room-s8": "BabyAI-OneRoomS8-v0",
    "babyai-open-door": "BabyAI-OpenDoor-v0",
    "babyai-open-doors-order-n4": "BabyAI-OpenDoorsOrderN4-v0",
    "babyai-open-red-door": "BabyAI-OpenRedDoor-v0",
    "babyai-open-two-doors": "BabyAI-OpenTwoDoors-v0",
    "babyai-open": "BabyAI-Open-v0",
    "babyai-pickup-above": "BabyAI-PickupAbove-v0",
    "babyai-pickup-dist": "BabyAI-PickupDist-v0",
    "babyai-pickup-loc": "BabyAI-PickupLoc-v0",
    "babyai-pickup": "BabyAI-Pickup-v0",
    "babyai-put-next-local": "BabyAI-PutNextLocal-v0",
    "babyai-put-next": "BabyAI-PutNextS7N4-v0",
    "babyai-synth-loc": "BabyAI-SynthLoc-v0",
    "babyai-synth-seq": "BabyAI-SynthSeq-v0",
    "babyai-synth": "BabyAI-Synth-v0",
    "babyai-unblock-pickup": "BabyAI-UnblockPickup-v0",
    "babyai-unlock-local": "BabyAI-UnlockLocal-v0",
    "babyai-unlock-pickup": "BabyAI-UnlockPickup-v0",
    "babyai-unlock-to-unlock": "BabyAI-UnlockToUnlock-v0",
    "babyai-unlock": "BabyAI-Unlock-v0",
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
