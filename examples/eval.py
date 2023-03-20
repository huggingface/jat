import os
from typing import List
from gia.config import Arguments
from gia.config.arguments import parse_args
from gia.eval.evaluator import Evaluator
from gia.eval.mujoco_evaluator import MujocoEvaluator
from gia.model.gia_model import GiaModel


EVAL_MAPPINGO = {
    "mujoco": MujocoEvaluator,
}


def create_evaluators(args: Arguments) -> List[Evaluator]:
    return [EVAL_MAPPINGO[task](args) for task in args.tasks]


def eval():
    args = parse_args()  # only need to specify the run directory
    args = Arguments.load_args(args)  # loads all the args from the run directory
    evaluators = create_evaluators(args)

    model = GiaModel(args)
    model = model.to("cuda")
    checkpoints_evaluated = set()  # TODO, iterate through checkpoints

    for evaluator in evaluators:
        result = evaluator.evaluate(model)


def list_files_in_dir(path):
    files = os.listdir(path)
    print()


if __name__ == "__main__":

    eval()
