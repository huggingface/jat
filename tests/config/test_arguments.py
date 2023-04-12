import os
import subprocess
import sys

import gia.config.arguments
from gia.config import Arguments


def test_command_line_options(tmp_path):
    # get the absolute path of gia.config.arguments.py
    path = os.path.abspath(gia.config.arguments.__file__)

    # Test setting some options via command line
    cmd = [
        sys.executable,
        path,
        "--save_dir",
        str(tmp_path),
        "--model_ckpt",
        "test_model",
        "--train_batch_size",
        "4",
        "--use_separator",
        "False",
        "--n_episodes",
        "5",
    ]

    subprocess.run(cmd, check=True)

    # Load the arguments from the file saved by the script
    loaded_args = Arguments()
    loaded_args.save_dir = str(tmp_path)
    loaded_args = Arguments.load_args(loaded_args)

    assert loaded_args.model_ckpt == "test_model"
    assert loaded_args.train_batch_size == 4
    assert loaded_args.use_separator is False
    assert loaded_args.n_episodes == 5

    # Test using a YAML config file
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(
            """
            save_dir: {}
            model_ckpt: yaml_test_model
            train_batch_size: 2
            """.format(
                tmp_path
            )
        )

    cmd = [sys.executable, path, str(config_path)]
    subprocess.run(cmd, check=True)

    # Load the arguments from the file saved by the script
    loaded_args = Arguments()
    loaded_args.save_dir = str(tmp_path)
    loaded_args = Arguments.load_args(loaded_args)

    assert loaded_args.model_ckpt == "yaml_test_model"
    assert loaded_args.train_batch_size == 2
