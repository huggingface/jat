import glob
import json
import subprocess

import wandb

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from accelerate import Accelerator
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from gia.config import Arguments
from gia.eval.utils import is_slurm_available


class EvaluateCheckpointCallback(TrainerCallback):
    EVAL_SLURM_SCRIPT = "scripts/cluster/launch_eval.slurm"

    def __init__(self) -> None:
        self._logged_files = set()

    # def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     wandb.init(id=args.wandb_run_id, group=args.wandb_run_group, project=args.wandb_project)
    #     # for custom x-axis on evals
    #     wandb.define_metric("eval/step")
    #     wandb.define_metric("eval/*", step_metric="eval/step")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not Accelerator().is_main_process:
            # otherwise multi-GPU jobs will launch several evals.
            return
        checkpoint_name = f"checkpoint-{state.global_step}"
        for task in args.task_names:
            self._launch_slurm_job(args, task, checkpoint_name)

        self._log_recent_evals_to_wandb(args)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # it is possible there is still an eval running that would not be logged to wandb
        # but the json file will still be on disk. TODO: improve this.
        self._log_recent_evals_to_wandb(args)

    def _launch_slurm_job(self, args: Arguments, task: str, checkpoint: str) -> None:
        launch_args = (
            f"--output_dir={args.output_dir} --task_names={task} "
            f"--eval_checkpoints={checkpoint} --n_episodes={args.n_episodes} "
            f"--wandb_run_id=train --wandb_project={args.wandb_project} "
            f"--wandb_run_group={args.wandb_run_group}"
        )
        if is_slurm_available():
            subprocess.call(["sbatch", self.EVAL_SLURM_SCRIPT, launch_args])
        else:
            print(launch_args)

    def _log_recent_evals_to_wandb(self, args: Arguments):
        # loads eval json files from disk and pushes them to wandb
        # I attempted to push this directly in the eval jobs but wandb
        # does not support parallel writing to the same report
        # I don't like that there are lots of small files here : TODO
        if not args.wandb_enabled:
            return

        try:  # I don't want this stopping training due to issues loading files etc
            all_eval_jsons = glob.glob(f"{args.output_dir}/eval/**/*.json", recursive=True)
            to_log = self._logged_files.difference(all_eval_jsons)

            for file in to_log:
                with open(file) as fp:
                    data = json.load(fp)

                task = data["task"]
                step = data["step"]
                result = data["result"]

                wandb_log_name = task.replace("-", "/")
                stats = {f"eval/{wandb_log_name}": result, "eval/step": step}

                wandb.log(stats)

                self._logged_files.add(file)

        except Exception as e:
            print("Exception logging evals to wandb:", e)
