import argparse
import os

import gymnasium as gym
import numpy as np
from bot_agent import Bot
from huggingface_hub import HfApi, repocard, upload_folder
from utils.env_wrappers import AddRGBImgPartialObsWrapper


def generate_dataset_card(
    dir_path: str,
    env: str,
    repo_id: str,
):
    readme_path = os.path.join(dir_path, "README.md")
    readme = f"""
    An imitation learning environment for the {env} environment. \n
    This environment was created as part of the Generally Intelligent Agents project gia:
    https://github.com/huggingface/gia \n
    \n
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {}
    metadata["library_name"] = "gia"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "gia",
        "multi-task",
        "multi-modal",
        "imitation-learning",
        "offline-reinforcement-learning",
    ]
    repocard.metadata_save(readme_path, metadata)


def push_to_hf(dir_path: str, repo_name: str):
    _ = HfApi().create_repo(repo_id=repo_name, private=False, exist_ok=True, repo_type="dataset")

    upload_folder(
        repo_id=repo_name, folder_path=dir_path, path_in_repo=".", ignore_patterns=[".git/*"], repo_type="dataset"
    )


def create_babyai_dataset(name_env, hf_repo_name, max_num_frames=100000, push_to_hub=False):
    env = gym.make(name_env)
    env = AddRGBImgPartialObsWrapper(env)  # add rgb image to obs

    class BabyAIEnvDataset:
        def __init__(self):
            self._data = {
                "text_observations": [],
                "discrete_observations": [],
                "discrete_actions": [],
                "rewards": []
            }

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

    dataset = BabyAIEnvDataset()

    def reset_env_and_policy(env):
        obs, _ = env.reset()
        policy = Bot(env.env)
        return obs, policy

    obs, policy = reset_env_and_policy(env)
    dataset.reset_episode()
    for i in range(max_num_frames):
        action = policy.replan()
        obs, r, done, _, infos = env.step(action)
        dataset.add_step(obs, int(action), r)
        if done:
            obs, policy = reset_env_and_policy(env)

        if done or i == max_num_frames - 1:
            dataset.reset_episode(append_new=i < max_num_frames - 1)

    env.close()

    if push_to_hub:
        repo_path = "./train_dir"
        os.makedirs(repo_path, exist_ok=True)

        with open(f"{repo_path}/dataset.npy", "wb") as f:
            np.save(f, dataset.to_dict())

        generate_dataset_card(repo_path, name_env, "")
        push_to_hf(repo_path, hf_repo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_env", type=str)
    parser.add_argument("--max_num_frames", default=100000, type=int)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_name", type=str)
    args = parser.parse_args()

    create_babyai_dataset(**vars(args))
