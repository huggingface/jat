#!/bin/bash
#SBATCH --job-name=create_babyai_dataset # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

#### CPU Partition ####
#SBATCH --account=imi@cpu
#SBATCH --qos=qos_cpu-t4

#SBATCH --array=0-39


NAMES=(
    babyai-action-obj-door
    babyai-blocked-unlock-pickup
    babyai-boss-level-no-unlock
    babyai-boss-level
    babyai-find-obj-s5
    babyai-go-to-door
    babyai-go-to-imp-unlock
    babyai-go-to-local
    babyai-go-to-obj-door
    babyai-go-to-obj
    babyai-go-to-red-ball-grey
    babyai-go-to-red-ball-no-dists
    babyai-go-to-red-ball
    babyai-go-to-red-blue-ball
    babyai-go-to-seq
    babyai-go-to
    babyai-key-corridor
    babyai-mini-boss-level
    babyai-move-two-across-s8n9
    babyai-one-room-s8
    babyai-open-door
    babyai-open-doors-order-n4
    babyai-open-red-door
    babyai-open-two-doors
    babyai-open
    babyai-pickup-above
    babyai-pickup-dist
    babyai-pickup-loc
    babyai-pickup
    babyai-put-next-local
    babyai-put-next
    babyai-synth-loc
    babyai-synth-seq
    babyai-synth
    babyai-unblock-pickup
    babyai-unlock-local
    babyai-unlock-pickup
    babyai-unlock-to-unlock
    babyai-unlock
)

module purge
module load python/3.8.2
conda activate hf_jat

python create_babyai_dataset.py --task_name=${NAMES[$SLURM_ARRAY_TASK_ID]} --max_num_episodes=100000