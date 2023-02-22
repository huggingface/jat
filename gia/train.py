import hydra

from gia.config.config import Config


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def train(config: Config):
    print(config)
    # training goes here


if __name__ == "__main__":
    train()
