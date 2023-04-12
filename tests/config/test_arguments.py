import json
import os
import subprocess
import sys

import gia.config.arguments
from gia.config import Arguments

EXEC_PATH = os.path.abspath(gia.config.arguments.__file__)


def test_save_args(tmp_path):
    # Test setting some options via command line
    args = Arguments(save_dir=str(tmp_path), model_ckpt="test_model", batch_size=4)
    args.save()
    assert (tmp_path / "args.json").exists()
    with open(tmp_path / "args.json") as f:
        loaded_args = json.load(f)
    assert loaded_args["model_ckpt"] == "test_model"
    assert loaded_args["batch_size"] == 4


def test_save_and_load_args(tmp_path):
    # Test setting some options via command line
    cmd = [
        sys.executable,
        EXEC_PATH,
        "--save_dir",
        str(tmp_path),
        "--model_ckpt",
        "test_model",
        "--batch_size",
        "4",
        "--use_separator",
        "False",
        "--n_episodes",
        "5",
    ]

    subprocess.run(cmd, check=True)

    # Load the arguments from the file saved by the script
    loaded_args = Arguments.load(tmp_path)

    assert loaded_args.model_ckpt == "test_model"
    assert loaded_args.batch_size == 4
    assert loaded_args.use_separator is False
    assert loaded_args.n_episodes == 5


def test_load_from_yaml_config_file(tmp_path):
    # Test using a YAML config file
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(
            f"""
save_dir: {tmp_path}
model_ckpt: yaml_test_model
batch_size: 2
"""
        )

    cmd = [sys.executable, EXEC_PATH, str(config_path)]
    subprocess.run(cmd, check=True)

    # Load the arguments from the file saved by the script
    loaded_args = Arguments.load(tmp_path)

    assert loaded_args.model_ckpt == "yaml_test_model"
    assert loaded_args.batch_size == 2
