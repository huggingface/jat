import os
from typing import List
from gia.config import Arguments
from .evaluator import Evaluator


def create_evaluators(args: Arguments) -> List[Evaluator]:
    return []


def eval(args: Arguments):
    evaluators = create_evaluators(args)

    checkpoints_evaluated = set()

    for evaluator in evaluators:
        result = evaluator.evaluate()


def list_files_in_dir(path):
    files = os.listdir(path)
    print()


if __name__ == "__main__":
    args = Arguments()
    args.load_from_path(args.checkpoint_dir)
    eval()
