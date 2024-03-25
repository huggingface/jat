"""
This script donwload the expert score from the dataset for all the
metaworld environments and saves them in a dictionary.
"""

import json
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


ENV_NAMES = [
    "atari-alien",
    "atari-amidar",
    "atari-assault",
    "atari-asterix",
    "atari-asteroids",
    "atari-atlantis",
    "atari-bankheist",
    "atari-battlezone",
    "atari-beamrider",
    "atari-berzerk",
    "atari-bowling",
    "atari-boxing",
    "atari-breakout",
    "atari-centipede",
    "atari-choppercommand",
    "atari-crazyclimber",
    "atari-defender",
    "atari-demonattack",
    "atari-doubledunk",
    "atari-enduro",
    "atari-fishingderby",
    "atari-freeway",
    "atari-frostbite",
    "atari-gopher",
    "atari-gravitar",
    "atari-hero",
    "atari-icehockey",
    "atari-jamesbond",
    "atari-kangaroo",
    "atari-krull",
    "atari-kungfumaster",
    "atari-montezumarevenge",
    "atari-mspacman",
    "atari-namethisgame",
    "atari-phoenix",
    "atari-pitfall",
    "atari-pong",
    "atari-privateeye",
    "atari-qbert",
    "atari-riverraid",
    "atari-roadrunner",
    "atari-robotank",
    "atari-seaquest",
    "atari-skiing",
    "atari-solaris",
    "atari-spaceinvaders",
    "atari-stargunner",
    "atari-surround",
    "atari-tennis",
    "atari-timepilot",
    "atari-tutankham",
    "atari-upndown",
    "atari-venture",
    "atari-videopinball",
    "atari-wizardofwor",
    "atari-yarsrevenge",
    "atari-zaxxon",
    "babyai-action-obj-door",
    "babyai-blocked-unlock-pickup",
    "babyai-boss-level-no-unlock",
    "babyai-boss-level",
    "babyai-find-obj-s5",
    "babyai-go-to-door",
    "babyai-go-to-imp-unlock",
    "babyai-go-to-local",
    "babyai-go-to-obj-door",
    "babyai-go-to-obj",
    "babyai-go-to-red-ball-grey",
    "babyai-go-to-red-ball-no-dists",
    "babyai-go-to-red-ball",
    "babyai-go-to-red-blue-ball",
    "babyai-go-to-seq",
    "babyai-go-to",
    "babyai-key-corridor",
    "babyai-mini-boss-level",
    "babyai-move-two-across-s8n9",
    "babyai-one-room-s8",
    "babyai-open-door",
    "babyai-open-doors-order-n4",
    "babyai-open-red-door",
    "babyai-open-two-doors",
    "babyai-open",
    "babyai-pickup-above",
    "babyai-pickup-dist",
    "babyai-pickup-loc",
    "babyai-pickup",
    "babyai-put-next-local",
    "babyai-put-next",
    "babyai-synth-loc",
    "babyai-synth-seq",
    "babyai-synth",
    "babyai-unblock-pickup",
    "babyai-unlock-local",
    "babyai-unlock-pickup",
    "babyai-unlock-to-unlock",
    "babyai-unlock",
    "metaworld-assembly",
    "metaworld-basketball",
    "metaworld-bin-picking",
    "metaworld-box-close",
    "metaworld-button-press-topdown-wall",
    "metaworld-button-press-topdown",
    "metaworld-button-press-wall",
    "metaworld-button-press",
    "metaworld-coffee-button",
    "metaworld-coffee-pull",
    "metaworld-coffee-push",
    "metaworld-dial-turn",
    "metaworld-disassemble",
    "metaworld-door-close",
    "metaworld-door-lock",
    "metaworld-door-open",
    "metaworld-door-unlock",
    "metaworld-drawer-close",
    "metaworld-drawer-open",
    "metaworld-faucet-close",
    "metaworld-faucet-open",
    "metaworld-hammer",
    "metaworld-hand-insert",
    "metaworld-handle-press-side",
    "metaworld-handle-press",
    "metaworld-handle-pull-side",
    "metaworld-handle-pull",
    "metaworld-lever-pull",
    "metaworld-peg-insert-side",
    "metaworld-peg-unplug-side",
    "metaworld-pick-out-of-hole",
    "metaworld-pick-place-wall",
    "metaworld-pick-place",
    "metaworld-plate-slide-back-side",
    "metaworld-plate-slide-back",
    "metaworld-plate-slide-side",
    "metaworld-plate-slide",
    "metaworld-push-back",
    "metaworld-push-wall",
    "metaworld-push",
    "metaworld-reach-wall",
    "metaworld-reach",
    "metaworld-shelf-place",
    "metaworld-soccer",
    "metaworld-stick-pull",
    "metaworld-stick-push",
    "metaworld-sweep-into",
    "metaworld-sweep",
    "metaworld-window-close",
    "metaworld-window-open",
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

FILENAME = "jat/eval/rl/scores_dict.json"

for env_name in tqdm(ENV_NAMES):
    tqdm.write(f"Downloading expert scores for {env_name}")

    dataset = load_dataset("jat-project/jat-dataset", env_name)
    # Initialize the variables
    rewards = dataset["train"]["rewards"] + dataset["test"]["rewards"]
    episode_sum_rewards = [np.sum(r) for r in rewards]

    # Load the scores dictionary
    if not os.path.exists(FILENAME):
        scores_dict = {}
    else:
        with open(FILENAME, "r") as file:
            scores_dict = json.load(file)

    # Add the expert scores to the dictionary
    if env_name not in scores_dict:
        scores_dict[env_name] = {}
    scores_dict[env_name]["expert"] = {"mean": np.mean(episode_sum_rewards), "std": np.std(episode_sum_rewards)}

    # Save the dictionary to a file
    with open(FILENAME, "w") as file:
        scores_dict = {
            task: {agent: scores_dict[task][agent] for agent in sorted(scores_dict[task])}
            for task in sorted(scores_dict)
        }
        json.dump(scores_dict, file, indent=4)
