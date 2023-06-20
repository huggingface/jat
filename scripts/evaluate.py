import os
import torch
import wandb
from gia import GiaConfig, GiaModel
from gia.config import Arguments
from gia.eval.utils import EVALUATORS, get_domain


def main():
    args = Arguments.parse_args()
    # connect to (existing) wandb
    wandb.init()
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    config = GiaConfig.from_args(args)
    model = GiaModel(config).to("cuda")

    for checkpoint_path in args.eval_checkpoints:
        step = int(checkpoint_path.split("-")[-1])
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        model.eval()
        for task in args.task_names:
            domain = get_domain(task)
            evaluator_cls = EVALUATORS[domain]
            evaluator = evaluator_cls(args, task)
            result = evaluator.evaluate(model)

            # log to wandb
            wandb_log_name = task.replace("-", "/")
            stats = {f"eval/{wandb_log_name}": result, "eval/step": step}

            wandb.log(stats)


if __name__ == "__main__":
    main()
