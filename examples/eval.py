import os
import time
from typing import List

import torch

from gia.config import Arguments
from gia.config.arguments import parse_args
from gia.eval.evaluator import Evaluator
from gia.eval.mujoco_evaluator import MujocoEvaluator
from gia.model.gia_model import GiaModel

EVAL_MAPPING = {
    "mujoco": MujocoEvaluator,
}


def create_evaluators(args: Arguments) -> List[Evaluator]:
    return [EVAL_MAPPING[task](args) for task in args.tasks]


def eval():
    args = parse_args()  # only need to specify the run directory
    args = Arguments.load(args.save_dir)  # loads all the args from the run directory
    evaluators = create_evaluators(args)

    model = GiaModel(args)
    model = model.to("cuda")

    checkpoints_path = os.path.join(args.save_dir, "checkpoints")
    evaluated_checkpoints = set()

    results = {}

    while True:
        time.sleep(1)
        all_checkpoints = set(os.listdir(checkpoints_path))
        if len(all_checkpoints - evaluated_checkpoints) == 0:
            print("all checkpoints evaluated")
        for filename in all_checkpoints - evaluated_checkpoints:
            checkpoint_path = os.path.join(checkpoints_path, filename)
            # Log to wandb directly?
            results[checkpoint_path] = eval_checkpoint(checkpoint_path, evaluators, model)
            evaluated_checkpoints.add(filename)


def eval_checkpoint(checkpoint_path: str, evaluators: List[Evaluator], model: torch.nn.Module):
    print("Evaluating checkpoint at:", checkpoint_path)

    state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        results = {}
        for evaluator in evaluators:
            results[evaluator.task] = evaluator.evaluate(model)

    return results


def list_files_in_dir(path):
    # files = os.listdir(path)
    print()


if __name__ == "__main__":
    eval()
