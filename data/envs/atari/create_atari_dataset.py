import datasets
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfApi, upload_folder
from PIL import Image
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import visualize_policy_inputs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log
from sf_examples.envpool.atari.train_envpool_atari import parse_atari_args, register_atari_components


def push_to_hf(dir_path: str, repo_name: str):
    _ = HfApi().create_repo(repo_id=repo_name, private=False, exist_ok=True, repo_type="dataset")

    upload_folder(
        repo_id=repo_name, folder_path=dir_path, path_in_repo=".", ignore_patterns=[".git/*"], repo_type="dataset"
    )


# most of this function is redundant as it is copied from sample.enjoy.enjoy
def create_atari_dataset(cfg: Config):

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1
    cfg.env_agents = 1

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

    num_frames = 0

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    image_observations = []
    rewards = []
    discrete_actions = []
    ep_image_observations = []
    ep_rewards = []
    ep_discrete_actions = []

    with torch.no_grad():
        while num_frames < cfg.max_num_frames:
            obs["obs"] = obs["obs"][0]
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

            # store s in buffer
            ep_image_observations.append(Image.fromarray(np.transpose(obs["obs"][0].cpu().numpy(), (1, 2, 0))))

            obs, rew, terminated, truncated, infos = env.step([actions])

            done = make_dones(terminated, truncated).item()

            # store a,r, d in buffer
            ep_rewards.append(rew.item())
            ep_discrete_actions.append(actions.item())

            num_frames += 1

            if done:  # fictious done
                rnn_states[0] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)

                if infos[0]["terminated"].item():
                    image_observations.append(ep_image_observations)
                    discrete_actions.append(np.array(ep_discrete_actions).astype(np.int64))
                    rewards.append(np.array(ep_rewards).astype(np.float32))
                    ep_image_observations = []
                    ep_discrete_actions = []
                    ep_rewards = []

                    log.info(f"Episode rewards: {np.sum(rewards[-1]):.3f}")

    env.close()

    task = cfg.env.split("_")[1]
    # Fix task names (see see https://huggingface.co/datasets/jat-project/jat-dataset/discussions/21 to 24)
    task = "asteroids" if task == "asteroid" else task
    task = "kungfumaster" if task == "kongfumaster" else task
    task = "montezumarevenge" if task == "montezuma" else task
    task = "privateeye" if task == "privateye" else task
    d = {
        "image_observations": image_observations,
        "discrete_actions": discrete_actions,
        "rewards": rewards,
    }
    features = datasets.Features(
        {
            "image_observations": datasets.Sequence(datasets.Image()),
            "discrete_actions": datasets.Sequence(datasets.Value("int64")),
            "rewards": datasets.Sequence(datasets.Value("float32")),
        }
    )

    ds = [
        Dataset.from_dict({k: [v[idx]] for k, v in d.items()}, features=features)
        for idx in range(len(d["image_observations"]))
    ]
    dataset = concatenate_datasets(ds)
    dataset = dataset.train_test_split(test_size=0.1, writer_batch_size=1)
    HfApi().create_branch("jat-project/jat-dataset", branch="new_breakout", exist_ok=True, repo_type="dataset")
    dataset.push_to_hub(
        "jat-project/jat-dataset",
        config_name=f"atari-{task}",
        branch="new_breakout",
    )


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args(evaluation=True)

    status = create_atari_dataset(cfg)
    return status


if __name__ == "__main__":
    main()
