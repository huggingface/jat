import os
from typing import Dict, Optional

import gym
import metaworld  # noqa: F401
import numpy as np
import torch
from huggingface_hub import HfApi, repocard, upload_folder
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.cfg.arguments import (
    load_from_checkpoint,
    parse_full_cfg,
    parse_sf_args,
)
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config

ENV_NAMES = [
    "assembly-v2",
    "basketball-v2",
    "bin-picking-v2",
    "box-close-v2",
    "button-press-topdown-v2",
    "button-press-topdown-wall-v2",
    "button-press-v2",
    "button-press-wall-v2",
    "coffee-button-v2",
    "coffee-pull-v2",
    "coffee-push-v2",
    "dial-turn-v2",
    "disassemble-v2",
    "door-close-v2",
    "door-lock-v2",
    "door-open-v2",
    "door-unlock-v2",
    "drawer-close-v2",
    "drawer-open-v2",
    "faucet-close-v2",
    "faucet-open-v2",
    "hammer-v2",
    "hand-insert-v2",
    "handle-press-side-v2",
    "handle-press-v2",
    "handle-pull-side-v2",
    "handle-pull-v2",
    "lever-pull-v2",
    "peg-insert-side-v2",
    "peg-unplug-side-v2",
    "pick-out-of-hole-v2",
    "pick-place-v2",
    "pick-place-wall-v2",
    "plate-slide-back-side-v2",
    "plate-slide-back-v2",
    "plate-slide-side-v2",
    "plate-slide-v2",
    "push-back-v2",
    "push-v2",
    "push-wall-v2",
    "reach-v2",
    "reach-wall-v2",
    "shelf-place-v2",
    "soccer-v2",
    "stick-pull-v2",
    "stick-push-v2",
    "sweep-into-v2",
    "sweep-v2",
    "window-close-v2",
    "window-open-v2",
]


def make_custom_env(
    full_env_name: str,
    cfg: Optional[Dict] = None,
    env_config: Optional[Dict] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    return gym.make(full_env_name, render_mode=render_mode)


def generate_dataset_card(dir_path: str, env: str, experiment_name: str):
    readme_path = os.path.join(dir_path, "README.md")
    hf_repo_name = f"prj_gia_dataset_metaworld_{env}_1111".replace("-", "_")
    readme = f"""
An imitation learning environment for the {env} environment, sample for the policy {experiment_name} \n
This environment was created as part of the Generally Intelligent Agents
project gia: https://github.com/huggingface/gia \n
\n

## Load dataset

First, clone it with

```sh
git clone https://huggingface.co/datasets/qgallouedec/{hf_repo_name}
```

Then, load it with

```python
import numpy as np
dataset = np.load("{hf_repo_name}/dataset.npy", allow_pickle=True).item()
print(dataset.keys())  # dict_keys(['observations', 'actions', 'dones', 'rewards'])
```

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


def push_to_hf(dir_path: str, repo_name: str) -> None:
    HfApi().create_repo(repo_id=repo_name, private=False, exist_ok=True, repo_type="dataset")
    upload_folder(
        repo_id=repo_name, folder_path=dir_path, path_in_repo=".", ignore_patterns=[".git/*"], repo_type="dataset"
    )


# most of this function is redundant as it is copied from sample.enjoy.enjoy
def create_dataset(cfg: Config):
    cfg = load_from_checkpoint(cfg)
    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    cfg.num_envs = 1

    # Create environment
    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    # Load checkpoint
    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    # Reset environment
    observations, _ = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    # Create dataset
    dataset_size = 100_000
    dataset = {
        "observations": np.zeros(
            (dataset_size, *env.observation_space["obs"].shape), dtype=env.observation_space["obs"].dtype
        ),
        "actions": np.zeros((dataset_size, *env.action_space.shape), env.action_space.dtype),
        "dones": np.zeros((dataset_size,), bool),
        "rewards": np.zeros((dataset_size,), np.float32),
    }

    # Run the environment
    with torch.no_grad():
        num_timesteps = 0
        while num_timesteps < dataset_size:
            normalized_obs = prepare_and_normalize_obs(actor_critic, observations)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # Sample actions from the distribution by default
            action_distribution = actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

            # Actions shape should be [num_agents, num_actions] even if it's [1, 1]
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]
            dataset["observations"][num_timesteps] = observations["obs"].cpu().numpy()
            dataset["actions"][num_timesteps] = actions

            observations, rewards, terminated, truncated, _ = env.step(actions)
            dones = make_dones(terminated, truncated)

            dataset["dones"][num_timesteps] = dones
            dataset["rewards"][num_timesteps] = rewards

            num_timesteps += 1

            dones = dones.cpu().numpy()
            for agent_idx, done in enumerate(dones):
                if done:
                    rnn_states[agent_idx] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)

    env.close()

    # Save dataset
    repo_path = f"{cfg.train_dir}/datasets/{cfg.experiment}"
    os.makedirs(repo_path, exist_ok=True)
    with open(f"{repo_path}/dataset.npy", "wb") as f:
        np.save(f, dataset)

    # Create dataset card and push to HF
    generate_dataset_card(repo_path, cfg.env, cfg.experiment)
    hf_repo_name = f"qgallouedec/prj_gia_dataset_metaworld_{cfg.env}_1111".replace("-", "_")
    push_to_hf(repo_path, hf_repo_name)


def main() -> int:
    for env_name in ENV_NAMES:
        register_env(env_name, make_custom_env)
    parser, _ = parse_sf_args(argv=None, evaluation=True)
    cfg = parse_full_cfg(parser)
    status = create_dataset(cfg)
    return status


if __name__ == "__main__":
    main()
