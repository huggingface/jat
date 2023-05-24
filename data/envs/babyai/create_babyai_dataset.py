import argparse
import os
import random

import gymnasium as gym
import numpy as np
from bot_agent import Bot
from utils.env_wrappers import AddRGBImgPartialObsWrapper


def create_babyai_dataset(name_env, saving_path, max_num_frames=100000, test_set_percentage=5):
    env = gym.make(name_env)
    env = AddRGBImgPartialObsWrapper(env)  # add rgb image to obs

    class BabyAIEnvDataset:
        def __init__(self):
            self._data = {"text_observations": [], "discrete_observations": [], "discrete_actions": [], "rewards": []}

        def reset_episode(self, append_new=True):
            for _, sequence in self._data.items():
                if len(sequence) > 0:
                    sequence[-1] = np.array(sequence[-1])

                if append_new:
                    sequence.append([])

        def add_step(self, obs, act, rew):
            self._data["text_observations"][-1].append(obs["mission"])
            flattened_symbolic_obs = obs["image"].flatten()
            concatenated_discrete_obs = np.append(obs["direction"], flattened_symbolic_obs)
            self._data["discrete_observations"][-1].append(concatenated_discrete_obs)
            self._data["discrete_actions"][-1].append(act)
            self._data["rewards"][-1].append(rew)

        def to_dict(self):
            return {k: np.array(v) for k, v in self._data.items()}

        def __len__(self):
            return len(self._data["rewards"])

    dataset = BabyAIEnvDataset()

    def reset_env_and_policy(env):
        obs, _ = env.reset()
        policy = Bot(env.env)
        return obs, policy

    print("Starting trajectories generation")
    obs, policy = reset_env_and_policy(env)
    dataset.reset_episode()
    for i in range(max_num_frames):
        try:
            action = policy.replan()
            obs, r, done, _, infos = env.step(action)
            dataset.add_step(obs, int(action), r)
        except Exception:
            dataset.reset_episode(append_new=i < max_num_frames - 1)
            done = True

        if done:
            obs, policy = reset_env_and_policy(env)

        if done or i == max_num_frames - 1:
            dataset.reset_episode(append_new=i < max_num_frames - 1)

    env.close()
    print("Finished generation")

    print("Saving...")
    dataset_size = len(dataset)
    test_set_indices = random.sample(range(dataset_size), round(dataset_size * test_set_percentage / 100))
    train_set_indices = [idx for idx in range(dataset_size) if idx not in test_set_indices]
    dict_dataset = dataset.to_dict()
    train_dataset = {k: v[train_set_indices] for k, v in dict_dataset.items()}
    test_dataset = {k: v[test_set_indices] for k, v in dict_dataset.items()}

    os.makedirs(saving_path, exist_ok=True)
    np.savez_compressed(f"{saving_path}/train.npz", **train_dataset)
    np.savez_compressed(f"{saving_path}/test.npz", **test_dataset)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_env", type=str)
    parser.add_argument("--saving_path", type=str)
    parser.add_argument("--max_num_frames", default=100000, type=int)
    parser.add_argument("--test_set_percentage", default=5, type=int)
    args = parser.parse_args()

    if os.path.exists(f"{args.saving_path}/train.npz") and os.path.exists(f"{args.saving_path}/test.npz"):
        print(f"Generated trajectories seem to already exist in {args.saving_path}, skipping generation")
    else:
        create_babyai_dataset(**vars(args))
