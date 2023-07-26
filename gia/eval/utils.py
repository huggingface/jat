import subprocess

from gia.eval.evaluator import Evaluator
from gia.eval.language_modeling.language_modeling_evaluator import LanguageModelingEvaluator
from gia.eval.rl import GymEvaluator


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


EVALUATORS = {
    "mujoco": GymEvaluator,
    "atari": GymEvaluator,
    "oscar": LanguageModelingEvaluator,
    "ok-vqa": LanguageModelingEvaluator,
    "conceptual-captions": LanguageModelingEvaluator,
}


def get_evaluator(task_name: str) -> Evaluator:
    """
    Get the evaluator for a given task.

    Args:
        task_name (`str`):
            The task name.

    Raises:
        `ValueError`: If the task name is unknown.

    Returns:
        evaluator (`Evaluator`):
            The evaluator for the task.
    """
    for domain in EVALUATORS.keys():
        if task_name.startswith(domain):
            return EVALUATORS[domain]
    else:
        raise ValueError(f"Unknown task {task_name}")
