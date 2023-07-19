import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from tqdm import tqdm


def make_custom_env(
    full_env_name: str,
    cfg: Optional[Dict] = None,
    env_config: Optional[Dict] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    return gym.make(full_env_name, render_mode=render_mode)


# most of this function is redundant as it is copied from sample.enjoy.enjoy
def create_dataset(cfg: Config) -> None:
    cfg = load_from_checkpoint(cfg)
    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    cfg.num_envs = 1  # only support 1 env

    # Create environment
    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    # Load checkpoint
    policy_id = cfg.policy_index
    name_prefix = {"latest": "checkpoint", "best": "best"}[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    # Create dataset
    dataset: Dict[str, List[List[Any]]] = {
        "continuous_observations": [],  # [[s0, s1, s2, ..., sT-1], [s0, s1, ...]], # terminal observation not stored
        "continuous_actions": [],  # [[a0, a1, a2, ..., aT-1], [a0, a1, ...]],
        "rewards": [],  # [[r1, r2, r3, ...,   rT], [r1, r2, ...]],
    }

    # Reset environment
    observations, _ = env.reset()
    dones = [True]
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    # Run the environment
    dataset_size = 1_600_000 + 160_000
    progress_bar = tqdm(total=dataset_size)
    num_timesteps = 0
    with torch.no_grad():
        while num_timesteps < dataset_size or not dones[0]:
            for agent_idx, done in enumerate(dones):
                if done:
                    rnn_states[agent_idx] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                    for value in dataset.values():
                        value.append([])

            progress_bar.update(1)
            normalized_obs = prepare_and_normalize_obs(actor_critic, observations)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # Sample actions from the distribution by default
            action_distribution = actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

            # Actions shape should be [num_agents, num_actions] even if it's [1, 1]
            actions = preprocess_actions(env_info, actions)
            # Clamp actions to be in the range of the action space
            actions = np.clip(actions, env.action_space.low, env.action_space.high)
            rnn_states = policy_outputs["new_rnn_states"]
            dataset["continuous_observations"][-1].append(observations["obs"].cpu().numpy()[0])
            dataset["continuous_actions"][-1].append(actions[0])

            observations, rewards, terminated, truncated, _ = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

            dataset["rewards"][-1].append(rewards.cpu().numpy())

            num_timesteps += 1

    env.close()

    dataset["continuous_observations"] = np.array(
        [np.array(x, dtype=np.float32) for x in dataset["continuous_observations"]], dtype=object
    )
    dataset["continuous_actions"] = np.array(
        [np.array(x, dtype=np.float32) for x in dataset["continuous_actions"]], dtype=object
    )
    dataset["rewards"] = np.array([np.array(x, dtype=np.float32) for x in dataset["rewards"]], dtype=object)

    repo_path = f"datasets/{cfg.experiment[:-3]}"
    os.makedirs(repo_path, exist_ok=True)

    _dataset = {key: value[:16_000] for key, value in dataset.items()}
    file = f"{repo_path}/train"
    np.savez_compressed(f"{file}.npz", **_dataset)

    _dataset = {key: value[16_000:] for key, value in dataset.items()}
    file = f"{repo_path}/test"
    np.savez_compressed(f"{file}.npz", **_dataset)


def main() -> int:
    parser, _ = parse_sf_args(argv=None, evaluation=True)
    cfg = parse_full_cfg(parser)
    register_env(cfg.env, make_custom_env)
    status = create_dataset(cfg)
    return status


if __name__ == "__main__":
    main()
