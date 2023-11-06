import os
import sys

import pytest
import yaml

import gato.config.arguments
from gato.config import Arguments


EXEC_PATH = os.path.abspath(gato.config.arguments.__file__)


@pytest.fixture(autouse=True)
def cleanup_argv():
    """Clean up sys.argv after each test to avoid side effects."""
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv


# def test_save(tmp_path):
#     args = Arguments(output_dir=str(tmp_path))
#     args.save()
#     out_path = Path(tmp_path) / "args.json"
#     assert out_path.exists()

#     with open(out_path, "r") as infile:
#         saved_args = json.load(infile)

#     assert saved_args == args.__dict__


# def test_save_and_load(tmp_path):
#     args = Arguments(output_dir=str(tmp_path))
#     args.save()

#     loaded_args = Arguments.load(str(tmp_path))
#     assert loaded_args.__dict__ == args.__dict__


# def test_command_line(tmp_path):
#     # Test setting some options via command line
#     cmd = [
#         sys.executable,
#         EXEC_PATH,
#         "--output_dir",
#         str(tmp_path),
#         "--per_device_train_batch_size",
#         "4",
#         "--use_separator",
#         "False",
#         "--n_episodes",
#         "5",
#     ]

#     subprocess.run(cmd, check=True)

#     # Load the arguments from the file saved by the script
#     loaded_args = Arguments.load(tmp_path)

#     # assert loaded_args.model_ckpt == "test_model"
#     assert loaded_args.per_device_train_batch_size == 4
#     assert loaded_args.use_separator is False
#     assert loaded_args.n_episodes == 5


# def test_load_from_yaml_config_file(tmp_path):
#     # Test using a YAML config file
#     config_path = tmp_path / "config.yaml"
#     with open(config_path, "w") as f:
#         f.write(
#             f"""
# output_dir: {tmp_path}
# per_device_train_batch_size: 2
# """
#         )

#     cmd = [sys.executable, EXEC_PATH, str(config_path)]
#     subprocess.run(cmd, check=True)

#     # Load the arguments from the file saved by the script
#     loaded_args = Arguments.load(tmp_path)

#     assert loaded_args.per_device_train_batch_size == 2


def test_unknown_task(tmp_path):
    with pytest.raises(ValueError):
        Arguments(task_names="unknown_task", output_dir=tmp_path)


def test_post_init_task_names_multi_domain(tmp_path):
    args = Arguments(task_names="mujoco,atari", output_dir=tmp_path)
    assert {*args.task_names} == {
        "atari-alien",
        "atari-amidar",
        "atari-assault",
        "atari-asterix",
        "atari-asteroids",
        "atari-atlantis",
        "atari-bankheist",
        "atari-battlezone",
        "atari-beamrider",
        "atari-berzerk",
        "atari-bowling",
        "atari-boxing",
        "atari-breakout",
        "atari-centipede",
        "atari-choppercommand",
        "atari-crazyclimber",
        "atari-defender",
        "atari-demonattack",
        "atari-doubledunk",
        "atari-enduro",
        "atari-fishingderby",
        "atari-freeway",
        "atari-frostbite",
        "atari-gopher",
        "atari-gravitar",
        "atari-hero",
        "atari-icehockey",
        "atari-jamesbond",
        "atari-kangaroo",
        "atari-krull",
        "atari-kungfumaster",
        "atari-montezumarevenge",
        "atari-mspacman",
        "atari-namethisgame",
        "atari-phoenix",
        "atari-pitfall",
        "atari-pong",
        "atari-privateeye",
        "atari-qbert",
        "atari-riverraid",
        "atari-roadrunner",
        "atari-robotank",
        "atari-seaquest",
        "atari-skiing",
        "atari-solaris",
        "atari-spaceinvaders",
        "atari-stargunner",
        "atari-surround",
        "atari-tennis",
        "atari-timepilot",
        "atari-tutankham",
        "atari-upndown",
        "atari-venture",
        "atari-videopinball",
        "atari-wizardofwor",
        "atari-yarsrevenge",
        "atari-zaxxon",
        "mujoco-ant",
        "mujoco-doublependulum",
        "mujoco-halfcheetah",
        "mujoco-hopper",
        "mujoco-humanoid",
        "mujoco-pendulum",
        "mujoco-pusher",
        "mujoco-reacher",
        "mujoco-standup",
        "mujoco-swimmer",
        "mujoco-walker",
    }


def test_babyai_go_to(tmp_path):
    args = Arguments(task_names="babyai-go-to", output_dir=tmp_path)
    assert {*args.task_names} == {"babyai-go-to"}


def test_post_init_task_names(tmp_path):
    args = Arguments(task_names="mujoco,atari-pong", output_dir=tmp_path)
    assert {*args.task_names} == {
        "atari-pong",
        "mujoco-ant",
        "mujoco-doublependulum",
        "mujoco-halfcheetah",
        "mujoco-hopper",
        "mujoco-humanoid",
        "mujoco-pendulum",
        "mujoco-pusher",
        "mujoco-reacher",
        "mujoco-standup",
        "mujoco-swimmer",
        "mujoco-walker",
    }


def test_parse_args_no_arguments(tmp_path):
    # Simulate no additional arguments apart from output dir which is required
    sys.argv = [EXEC_PATH, f"--output_dir={tmp_path}"]

    args = Arguments.parse_args()

    assert len(args.task_names) > 150  # ensure that there are many tasks (currently there are 151)


def test_parse_args_custom_arguments(tmp_path):
    # Simulate custom arguments
    sys.argv = [EXEC_PATH, f"--output_dir={tmp_path}", "--task_names=mujoco,atari-pong"]

    args = Arguments.parse_args()

    assert args.output_dir == str(tmp_path)
    assert {*args.task_names} == {
        "atari-pong",
        "mujoco-ant",
        "mujoco-doublependulum",
        "mujoco-halfcheetah",
        "mujoco-hopper",
        "mujoco-humanoid",
        "mujoco-pendulum",
        "mujoco-pusher",
        "mujoco-reacher",
        "mujoco-standup",
        "mujoco-swimmer",
        "mujoco-walker",
    }


def test_parse_args_yaml_file(tmp_path):
    # Create a temporary YAML file for testing
    yaml_file_path = os.path.join(tmp_path, "config.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump({"output_dir": str(tmp_path), "task_names": ["mujoco", "atari-pong"]}, yaml_file)

    # Simulate passing a YAML file as an argument
    sys.argv = [EXEC_PATH, yaml_file_path]

    args = Arguments.parse_args()

    assert args.output_dir == str(tmp_path)
    assert {*args.task_names} == {
        "atari-pong",
        "mujoco-ant",
        "mujoco-doublependulum",
        "mujoco-halfcheetah",
        "mujoco-hopper",
        "mujoco-humanoid",
        "mujoco-pendulum",
        "mujoco-pusher",
        "mujoco-reacher",
        "mujoco-standup",
        "mujoco-swimmer",
        "mujoco-walker",
    }
