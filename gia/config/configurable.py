from gia.config.config import Config


class Configurable:
    def __init__(self, config: Config):
        self.config: Config = config
