#!/bin/bash
#SBATCH --job-name=gia_proc   # name of job
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err

source /fsx/ed/miniconda3/etc/profile.d/conda.sh

conda activate gia

TASKS=$1

python scripts/tokenize_dataset.py --output_dir=runs/proc_dataset --task_names=$TASKS --preprocessing_num_worker=16