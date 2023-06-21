import wandb
import subprocess

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from gia.config import Arguments
from gia.eval.utils import is_slurm_available


class EvaluateCheckpointCallback(TrainerCallback):
    EVAL_SLURM_SCRIPT = "scripts/cluster/launch_eval.slurm"

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print("on save callback", args, state, control, wandb.run.id, kwargs)
        # return super().on_save(args, state, control, **kwargs)
        checkpoint_name = f"checkpoint-{state.global_step}"
        for task in args.task_names:
            self._launch_slurm_job(args, task, checkpoint_name)

    def _launch_slurm_job(self, args: Arguments, task: str, checkpoint: str) -> None:
        launch_args = (
            f"--output_dir={args.output_dir} --task_names={task} --eval_checkpoints={checkpoint} --n_episodes={args.n_episodes}"
            f"--wandb_run_id={wandb.run.id} --wandb_project={args.wandb_project} --wandb_run_group={args.wandb_run_group}"
        )
        if is_slurm_available():
            subprocess.call(["sbatch", self.EVAL_SLURM_SCRIPT, launch_args])
        else:
            print(launch_args)
