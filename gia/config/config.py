from dataclasses import dataclass, field
from typing import List, Tuple
from hydra import initialize, compose

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


@dataclass
class Paths:
    output_dir: str = "outputs/default"


@dataclass
class Model:
    encoder_conv_architecture: str = "convnet_simple"
    nonlinearity: str = "relu"
    encoder_conv_mlp_layers: List[int] = field(default_factory=lambda: [512])
    encoder_mlp_layers: List[int] = field(default_factory=lambda: [512, 512])

    actor_critic_share_weights: bool = True
    use_rnn: bool = False
    decoder_mlp_layers: List[int] = field(default_factory=lambda: [])
    adaptive_stddev: bool = True
    policy_init_gain: float = 1.0
    policy_initialization: str = "orthogonal"


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
    paths: Paths
    model: Model
    obs: Obs


# useful for testing
def get_config() -> Config:
    with initialize(version_base=None, config_path="../config"):
        config = compose(
            config_name="config",
        )
        return config


print(get_config())
