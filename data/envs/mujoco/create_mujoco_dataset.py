import os
import time
from collections import deque

import numpy as np
import torch
from huggingface_hub import HfApi, repocard, upload_folder
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import render_frame, visualize_policy_inputs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log
from sf_examples.mujoco.train_mujoco import parse_mujoco_cfg, register_mujoco_components

from data.to_hub import add_dataset_to_hub


def generate_dataset_card(
    dir_path: str,
    env: str,
    experiment_name: str,
    repo_id: str,
):
    readme_path = os.path.join(dir_path, "README.md")
    readme = f"""
    An imitation learning environment for the {env} environment, sample for the policy {experiment_name} \n
    This environment was created as part of the Jack of All Trades (JAT) project:
    https://github.com/huggingface/jat \n
    \n
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {}
    metadata["library_name"] = "jat"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "jat",
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


# most of this function is redundant as it is copied from sample.enjoy.enjoy
def create_mujoco_dataset(cfg: Config):
    verbose = False

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = {"latest": "checkpoint", "best": "best"}[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0

    # class AtariEnvDataset:
    #     def __init__(self, num_frames):
    #         self.observations = np.zeros((num_frames, *obs["obs"].shape), dtype=np.float32)
    #         self.actions = np.zeros((num_frames, env.action_space.shape[0]))
    #         self.dones = np.zeros((num_frames, 1))  # bool
    #         self.rewards = np.zeros((num_frames, 1))  # float32

    #     def to_dict(self):
    #         return {
    #             "observations": self.observations,
    #             "actions": self.actions,
    #             "dones": self.dones,
    #             "rewards": self.rewards,
    #         }

    # dataset = AtariEnvDataset(cfg.max_num_frames)

    dataset_continuous_observations = []
    dataset_rewards = []
    dataset_continuous_actions = []
    ep_continuous_observations = []
    ep_rewards = []
    ep_continuous_actions = []

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):  # this is 1 for all atari envs
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                # store s in buffer
                if num_frames < cfg.max_num_frames:
                    ep_continuous_observations.append(obs["obs"].cpu().numpy())
                    # dataset.observations[num_frames] = obs["obs"].cpu().numpy()

                obs, rew, terminated, truncated, infos = env.step(actions)

                dones = make_dones(terminated, truncated)

                # store a,r, d in buffer
                if num_frames < cfg.max_num_frames:
                    ep_rewards.append(rew[0])
                    ep_continuous_actions.append(actions)

                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death
                            # (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):  # only 1 env
                    dataset_continuous_observations.append(
                        np.squeeze(np.array(ep_continuous_observations).astype(np.float32), axis=1)
                    )
                    dataset_continuous_actions.append(
                        np.squeeze(np.array(ep_continuous_actions).astype(np.float32), axis=1)
                    )
                    dataset_rewards.append(np.array(ep_rewards).astype(np.float32))
                    ep_continuous_observations = []
                    ep_continuous_actions = []
                    ep_rewards = []

                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                    )

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    task = cfg.env.split("_")[1]
    add_dataset_to_hub(
        "mujoco",
        task,
        continuous_observations=dataset_continuous_observations,
        continuous_actions=dataset_continuous_actions,
        rewards=dataset_rewards,
        push_to_hub=cfg.push_to_hub,
        revision=task,
    )


def main():
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_mujoco_cfg(evaluation=True)

    status = create_mujoco_dataset(cfg)
    return status


if __name__ == "__main__":
    main()
