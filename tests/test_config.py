from hydra import compose, initialize


def test_load_config():
    with initialize(version_base=None, config_path="../gia/config"):
        config = compose(
            config_name="config",
        )
        print(config)
