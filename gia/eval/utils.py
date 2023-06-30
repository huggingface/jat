import subprocess

from gia.eval.language_modeling.language_modeling_evaluator import LanguageModelingEvaluator
from gia.eval.rl import GymEvaluator


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


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
