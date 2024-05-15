#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
##SBATCH --gres=gpu:8
#SBATCH --output=/fsx/qgallouedec/logs/%x-%j.out
#SBATCH --err=/fsx/qgallouedec/logs/%x-%j.err

set -x -e
source ~/.bashrc
conda activate jat
export CPATH=$CONDA_PREFIX/include
accelerate launch scripts/train_jat_tokenized.py \
    --output_dir checkpoints/jat_small \
    --model_name_or_path jat-project/jat-small \
    --tasks atari babyai metaworld mujoco \
    --trust_remote_code \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --save_steps 10000 \
    --run_name train_jat_small \
    --logging_steps 100 \
    --logging_first_step \
    --dispatch_batches False \
    --remove_unused_columns False \
    --dataloader_num_workers 16 \
    --max_steps 250000