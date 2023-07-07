import sys
from typing import Dict, Optional

import gymnasium as gym
import metaworld  # noqa: F401
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import register_env


def make_custom_env(
    full_env_name: str,
    cfg: Optional[Dict] = None,
    env_config: Optional[Dict] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    return gym.make(full_env_name, render_mode=render_mode)


def main() -> int:
    parser, _ = parse_sf_args(argv=None, evaluation=True)
    cfg = parse_full_cfg(parser)
    register_env(cfg.env, make_custom_env)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
