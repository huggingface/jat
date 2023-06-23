import os
import json

import torch

from gia import GiaConfig, GiaModel
from gia.config import Arguments
from gia.eval.utils import get_evaluator


def main():
    args = Arguments.parse_args()
    config = GiaConfig.from_args(args)
    model = GiaModel(config).to("cuda")

    for checkpoint in args.eval_checkpoints:
        step = int(checkpoint.split("-")[-1])
        state_dict = torch.load(os.path.join(args.output_dir, checkpoint, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        model.eval()

        for task in args.task_names:
            evaluator_cls = get_evaluator(task)
            evaluator = evaluator_cls(args, task)
            result = evaluator.evaluate(model)
            # write to a json file

            data = {
                "checkpoint": checkpoint,
                "task": task,
                "step": step,
                "result": result,
            }

            output_path = os.path.join(args.output_dir, "evals", task, f"eval_{checkpoint}.json")
            with open(output_path) as fp:
                json.dump(data, fp)


if __name__ == "__main__":
    main()
