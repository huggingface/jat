import os

import torch
import wandb

from gia import GiaConfig, GiaModel
from gia.config import Arguments
from gia.eval.utils import get_evaluator


def main():
    args = Arguments.parse_args()
    # connect to (existing) wandb
    wandb.init(resume="must", id=args.wandb_run_id, group=args.wandb_run_group, project=args.wandb_project)
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

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

            # log to wandb
            wandb_log_name = task.replace("-", "/")
            stats = {f"eval/{wandb_log_name}": result, "eval/step": step}

            wandb.log(stats)


if __name__ == "__main__":
    main()
