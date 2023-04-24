import argparse
import sys
from typing import Dict, Optional

import gym
import metaworld  # noqa: F401
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

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


def override_defaults(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(
        batched_sampling=False,
        device="cpu",
        num_workers=8,
        num_envs_per_worker=8,
        worker_num_splits=2,
        train_for_env_steps=100_000_000,
        encoder_mlp_layers=[64, 64],
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=1024,
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        restart_behavior="restart",
        reward_scale=0.1,
        rollout=64,
        max_grad_norm=3.5,
        num_epochs=2,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=1.3,
        exploration_loss_coeff=0.0,
        learning_rate=0.00295,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=3,
        save_every_sec=15,
        serial_mode=False,
        async_rl=False,
    )
    return parser


def main() -> int:
    for env_name in ENV_NAMES:
        register_env(env_name, make_custom_env)
    parser, _ = parse_sf_args(argv=None, evaluation=False)
    parser = override_defaults(parser)
    cfg = parse_full_cfg(parser)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
