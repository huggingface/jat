from dataclasses import dataclass, field
from typing import List

from hydra import compose, initialize


# The contents of this file should make gia/config/config.yaml
@dataclass
class Dist:
    n_learners: int = 1
    n_workers: int = 3
    world_size: int = 4


@dataclass
class Hyp:
    n_epochs: int = 4
    n_agents: int = 8
    mini_batch_size: int = 4
    rollout_length: int = 16
    lr: float = 1e-4


class RL:
    normalize_returns: bool = True


class Envs:
    agents_per_env: int = 64
    frameskip: int = 4


@dataclass
class Paths:
    output_dir: str = "outputs/default"


@dataclass
class Model:
    model_name = "facebook/opt-125m"


@dataclass
class Obs:
    subtract_mean: float = 0.0
    scale: float = 1.0
    normalize_input: bool = True
    normalize_input_keys: str = "*"


@dataclass
class Config:
    dist: Dist
    hyp: Hyp
    rl: RL
    envs: Envs
    paths: Paths
    model: Model
    obs: Obs

    # useful for testing
    @staticmethod
    def build() -> "Config":
        with initialize(version_base=None, config_path="../config"):
            config = compose(
                config_name="config",
            )
            return config


# print(Config.build())
