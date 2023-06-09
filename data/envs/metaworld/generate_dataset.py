import os
from typing import Any, Dict, List, Optional

import gym
import metaworld  # noqa: F401
import numpy as np
import torch
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


# most of this function is redundant as it is copied from sample.enjoy.enjoy
def create_dataset(cfg: Config, dataset_size: int = 100_000, split: str = "train") -> None:
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
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    for value in dataset:
        value.append([])

    # Run the environment
    dones = [False]
    with torch.no_grad():
        num_timesteps = 0
        while num_timesteps < dataset_size or not dones[0]:
            normalized_obs = prepare_and_normalize_obs(actor_critic, observations)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # Sample actions from the distribution by default
            action_distribution = actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

            # Actions shape should be [num_agents, num_actions] even if it's [1, 1]
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]
            dataset["continuous_observations"][-1].append(observations["obs"].cpu().numpy()[0])
            dataset["continuous_actions"][-1].append(actions[0])

            observations, rewards, terminated, truncated, _ = env.step(actions)
            dones = make_dones(terminated, truncated)

            dataset["rewards"][-1].append(rewards.cpu().numpy())

            num_timesteps += 1

            dones = dones.cpu().numpy()
            for agent_idx, done in enumerate(dones):
                if done:
                    rnn_states[agent_idx] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                    for value in dataset:
                        value.append([])

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
    file = f"{repo_path}/{split}"
    np.savez_compressed(f"{file}.npz", **dataset)


def main() -> int:
    for env_name in ENV_NAMES:
        register_env(env_name, make_custom_env)
    parser, _ = parse_sf_args(argv=None, evaluation=True)
    cfg = parse_full_cfg(parser)
    status = create_dataset(cfg, dataset_size=90_000, split="train")
    status = create_dataset(cfg, dataset_size=10_000, split="test")
    return status


if __name__ == "__main__":
    main()
