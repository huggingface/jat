import sys
from typing import Optional
from sample_factory.enjoy import enjoy

import metaworld
import gym

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

def parse_args(argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
#    add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
#    custom_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def make_custom_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
    # see the section below explaining arguments
    return gym.make("pick-place-v2")

def main():
    """Script entry point."""
    register_env("pick-place-v2", make_custom_env)
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())