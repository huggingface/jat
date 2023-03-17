import sys
from typing import Dict, Optional

import gym
import metaworld
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import register_env

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


def main() -> int:
    for env_name in ENV_NAMES:
        register_env(env_name, make_custom_env)
    parser, _ = parse_sf_args(argv=None, evaluation=True)
    cfg = parse_full_cfg(parser)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
