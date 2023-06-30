import os
import json

import torch

from transformers import AutoConfig, AutoModel
from gia.config import Arguments
from gia.eval.utils import get_evaluator


def main():
    args = Arguments.parse_args()
    config = AutoConfig.from_pretrained(
        args.config_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    model = AutoModel.from_config(config=config).to("cuda")

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
            os.makedirs(os.path.join(args.output_dir, "evals", task), exist_ok=True)
            output_path = os.path.join(args.output_dir, "evals", task, f"eval_{checkpoint}.json")
            with open(output_path, "w") as fp:
                json.dump(data, fp)


if __name__ == "__main__":
    main()
