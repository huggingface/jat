import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import gia.config.arguments
from gia.config import Arguments, parse_args

EXEC_PATH = os.path.abspath(gia.config.arguments.__file__)


@pytest.fixture(autouse=True)
def cleanup_argv():
    """Clean up sys.argv after each test to avoid side effects."""
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv


def test_save(tmp_path):
    args = Arguments(output_dir=str(tmp_path))
    args.save()
    out_path = Path(tmp_path) / "args.json"
    assert out_path.exists()

    with open(out_path, "r") as infile:
        saved_args = json.load(infile)

    assert saved_args == args.__dict__


def test_save_and_load(tmp_path):
    args = Arguments(save_dir=str(tmp_path))
    args.save()

    loaded_args = Arguments.load(str(tmp_path))
    assert loaded_args.__dict__ == args.__dict__


def test_command_line(tmp_path):
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


def test_generate_save_dir():
    base_dir = "./runs/"
    os.makedirs(base_dir, exist_ok=True)
    existing_run_dirs = [d.name for d in Path(base_dir).iterdir() if d.is_dir()]
    run_indices = [int(match.group(1)) for run_dir in existing_run_dirs if (match := re.match(r"run_(\d+)", run_dir))]
    highest_idx = max(run_indices, default=0)

    generated_save_dir = Arguments._generate_save_dir()
    assert generated_save_dir == f"{base_dir}run_{highest_idx + 1}"


def test_post_init_default_save_dir():
    args = Arguments()
    assert args.save_dir.startswith("./runs/run_")


def test_post_init_task_names():
    args = Arguments(task_names="task1,task2,task3")
    assert args.task_names == ["task1", "task2", "task3"]


def test_parse_args_no_arguments():
    # Simulate no additional arguments
    sys.argv = [EXEC_PATH]

    args = parse_args()

    assert args.save_dir.startswith("./runs/run_")
    assert args.task_names == ["all"]


def test_parse_args_custom_arguments(tmp_path):
    # Simulate custom arguments
    sys.argv = [EXEC_PATH, f"--save_dir={tmp_path}", "--task_names=task1,task2"]

    args = parse_args()

    assert args.save_dir == str(tmp_path)
    assert args.task_names == ["task1", "task2"]


def test_parse_args_yaml_file(tmp_path):
    # Create a temporary YAML file for testing
    yaml_file_path = os.path.join(tmp_path, "config.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump({"save_dir": str(tmp_path), "task_names": ["task1", "task2"]}, yaml_file)

    # Simulate passing a YAML file as an argument
    sys.argv = [EXEC_PATH, yaml_file_path]

    args = parse_args()

    assert args.save_dir == str(tmp_path)
    assert args.task_names == ["task1", "task2"]
