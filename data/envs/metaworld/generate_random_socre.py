import json
import multiprocessing as mp

import gym
import metaworld  # noqa: F401
import numpy as np
from uncertainties import ufloat

from gia.utils import ufloat_decoder, ufloat_encoder

N_EPISODES = 1000

ENV_IDS = [
    "assembly",
    "basketball",
    "bin-picking",
    "box-close",
    "button-press-topdown",
    "button-press-topdown-wall",
    "button-press",
    "button-press-wall",
    "coffee-button",
    "coffee-pull",
    "coffee-push",
    "dial-turn",
    "disassemble",
    "door-close",
    "door-lock",
    "door-open",
    "door-unlock",
    "drawer-close",
    "drawer-open",
    "faucet-close",
    "faucet-open",
    "hammer",
    "hand-insert",
    "handle-press-side",
    "handle-press",
    "handle-pull-side",
    "handle-pull",
    "lever-pull",
    "peg-insert-side",
    "peg-unplug-side",
    "pick-out-of-hole",
    "pick-place",
    "pick-place-wall",
    "plate-slide-back-side",
    "plate-slide-back",
    "plate-slide-side",
    "plate-slide",
    "push-back",
    "push",
    "push-wall",
    "reach",
    "reach-wall",
    "shelf-place",
    "soccer",
    "stick-pull",
    "stick-push",
    "sweep-into",
    "sweep",
    "window-close",
    "window-open",
]


def get_random_score(env_id):
    env = gym.make(f"{env_id}-v2")
    env.reset()
    num_timesteps = 0
    tot_rewards = []
    tot_reward = 0
    successes = []
    while num_timesteps < N_EPISODES * 500:  # All episodes are 500 timesteps
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        assert info["success"] == info["is_success"]
        tot_reward += reward
        if terminated or truncated:
            env.reset()
            tot_rewards.append(tot_reward)
            tot_reward = 0
            successes.append(info["success"])

        num_timesteps += 1
    env.close()
    return ufloat(np.mean(tot_rewards), np.std(tot_rewards)), ufloat(np.mean(successes), np.std(successes))


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        scores_successes = pool.map(get_random_score, ENV_IDS)

    scores = {env_id: {"random": score[0]} for env_id, score in zip(ENV_IDS, scores_successes)}
    sucesses = {env_id: {"random": score[1]} for env_id, score in zip(ENV_IDS, scores_successes)}

    # Saving
    with open("data/envs/metaworld/scores.json", "w") as f:
        json.dump(scores, f, default=ufloat_encoder, indent=4)

    with open("data/envs/metaworld/successes.json", "w") as f:
        json.dump(sucesses, f, default=ufloat_encoder, indent=4)

    # Loading
    with open("data/envs/metaworld/scores.json", "r") as f:
        json.load(f, object_hook=ufloat_decoder)

    with open("data/envs/metaworld/successes.json", "r") as f:
        json.load(f, object_hook=ufloat_decoder)


if __name__ == "__main__":
    main()
