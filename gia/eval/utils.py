import subprocess

from gia.config.arguments import Arguments
from gia.eval.language_modeling.language_modeling_evaluator import LanguageModelingEvaluator
from gia.eval.rl import GymEvaluator


def is_slurm_available() -> bool:
    return True
    # returns true if a slurm queueing system is available
    try:
        result = subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def launch_evaluation(args: Arguments, checkpoint_path: str):
    assert is_slurm_available()

    for task in args.task_names:
        pass
        # launch eval job


# TODO: A nice use case for structural pattern matching?!
EVALUATORS = {
    "mujoco": GymEvaluator,
    "atari": GymEvaluator,
    "oscar": LanguageModelingEvaluator,
    "ok-vqa": LanguageModelingEvaluator,
    "conceptual-captions": LanguageModelingEvaluator,
}


def get_evaluator(task):
    if "-" in task:
        domain = task.split("-")[0]  # TODO: this will have problems for ok-vqa, etc..
        return EVALUATORS[domain]

    return EVALUATORS[task]
