import pytest
import hydra
from hydra import initialize, compose
from gia.config.config import Config


def test_load_config():
    with initialize(version_base=None, config_path="../gia/config"):
        config = compose(
            config_name="config",
        )
        print(config)
