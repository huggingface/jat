#!/bin/bash
#SBATCH --job-name=gia_eval     # name of job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err

source /fsx/ed/miniconda3/etc/profile.d/conda.sh

conda activate gia
echo $1
python /fsx/ed/work/gia/scripts/evaluate.py $1