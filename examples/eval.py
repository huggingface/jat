import os
import time
from typing import List

import torch

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
    args: Arguments = Arguments.load_args(args)  # loads all the args from the run directory
    evaluators = create_evaluators(args)

    model = GiaModel(args)
    model = model.to("cuda")

    checkpoint_path = os.path.join(args.save_dir, "checkpoints")

    current_checkpoints = set(os.listdir(checkpoint_path))

    results = {}

    while True:
        time.sleep(1)
        new_checkpoints = set(os.listdir(checkpoint_path))

        for filename in new_checkpoints - current_checkpoints:
            checkpoint_path = os.path.join(checkpoint_path, filename)
            # Log to wandb directly?
            results[checkpoint_path] = eval_checkpoint(checkpoint_path, evaluators)


def eval_checkpoint(checkpoint_path: str, evaluators: List[Evaluator], model: torch.nn.Module):
    print(checkpoint_path)
    return
    state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        results = {}

        for evaluator in evaluators:
            results[evaluator.task] = evaluator.evaluate(model)

    return results


def list_files_in_dir(path):
    files = os.listdir(path)
    print()


if __name__ == "__main__":

    eval()
