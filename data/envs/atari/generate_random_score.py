import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sf_examples.atari.train_atari import parse_atari_args, register_atari_components


FILENAME = "jat/eval/rl/scores_dict.json"


TASK_NAMES = [
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
]

TOT_NUM_EPISODES = 100


def generate_random_score(task_name):
    cfg = parse_atari_args(evaluation=True)
    env_id = task_name.replace("-", "_")
    if env_id == "atari_asteroids":
        env_id = "atari_asteroid"
    if env_id == "atari_montezumarevenge":
        env_id = "atari_montezuma"
    if env_id == "atari_kungfumaster":
        env_id = "atari_kongfumaster"
    if env_id == "atari_privateeye":
        env_id = "atari_privateye"
    cfg.env = env_id
    eval_env_frameskip = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    cfg.num_envs = 16

    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))

    ep_rewards = []
    env.reset()

    with torch.no_grad():
        while len(ep_rewards) < TOT_NUM_EPISODES:
            _, _, _, _, infos = env.step(np.array([env.action_space.sample() for _ in range(cfg.num_envs)]))

            for info in infos:
                if "episode" in info:
                    ep_rewards.append(info["episode"]["r"][0])
                    if len(ep_rewards) % 10 == 0:
                        print(f"Task {task_name} - progress {int(len(ep_rewards) / TOT_NUM_EPISODES * 100)}%")

    env.close()

    # Load the scores dictionary
    if not os.path.exists(FILENAME):
        scores_dict = {}
    else:
        with open(FILENAME, "r") as file:
            scores_dict = json.load(file)

    # Add the random scores to the dictionary
    if task_name not in scores_dict:
        scores_dict[task_name] = {}
    scores_dict[task_name]["random"] = {"mean": float(np.mean(ep_rewards)), "std": float(np.std(ep_rewards))}

    # Save the dictionary to a file
    with open(FILENAME, "w") as file:
        scores_dict = {
            task: {agent: scores_dict[task][agent] for agent in sorted(scores_dict[task])}
            for task in sorted(scores_dict)
        }
        json.dump(scores_dict, file, indent=4)


if __name__ == "__main__":
    register_atari_components()
    with Pool(32) as p:
        p.map(generate_random_score, TASK_NAMES)
