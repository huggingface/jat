#!/bin/bash
#SBATCH --job-name=gia_train     # name of job
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:a100:8
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err

source /fsx/ed/miniconda3/etc/profile.d/conda.sh

conda activate gia

accelerate launch /fsx/ed/work/gia/scripts/train.py --output_dir=runs/run_69 --task_names=mujoco --save_steps=100 --logging_steps=10 \
--preprocessing_num_worker=24 --wandb_project=gia --wandb_run_group=train_20230623_mujoco_test_eval5 --wandb_run_id=train --max_steps=1000
