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

#SBATCH --array=0-37

ENVS=(
    BabyAI-GoTo-v0 BabyAI-GoToDoor-v0 BabyAI-GoToLocal-v0 BabyAI-GoToObj-v0 BabyAI-GoToObjDoor-v0 BabyAI-GoToRedBall-v0 BabyAI-GoToRedBallGrey-v0 BabyAI-GoToRedBallNoDists-v0 BabyAI-GoToRedBlueBall-v0 BabyAI-GoToSeq-v0 BabyAI-Open-v0 BabyAI-OpenDoor-v0 BabyAI-OpenDoorsOrderN4-v0 BabyAI-OpenRedDoor-v0 BabyAI-OpenTwoDoors-v0 BabyAI-ActionObjDoor-v0 BabyAI-FindObjS5-v0 BabyAI-KeyCorridor-v0 BabyAI-MoveTwoAcrossS8N9-v0 BabyAI-OneRoomS8-v0 BabyAI-Pickup-v0 BabyAI-PickupAbove-v0 BabyAI-PickupDist-v0 BabyAI-PickupLoc-v0 BabyAI-UnblockPickup-v0 BabyAI-BossLevel-v0 BabyAI-BossLevelNoUnlock-v0 BabyAI-MiniBossLevel-v0 BabyAI-Synth-v0 BabyAI-SynthLoc-v0 BabyAI-SynthSeq-v0 BabyAI-BlockedUnlockPickup-v0 BabyAI-UnlockLocal-v0 BabyAI-UnlockPickup-v0
)

NAMES=(
    go-to go-to-door go-to-local go-to-obj go-to-obj-door go-to-red-ball go-to-red-ball-grey go-to-red-ball-no-dists go-to-red-blue-ball go-to-seq open open-door open-doors-order-n4 open-red-door open-two-doors action-obj-door find-obj-s5 key-corridor move-two-across-s8n9 one-room-s8 pickup pickup-above pickup-dist pickup-loc unblock-pickup boss-level boss-level-no-unlock mini-boss-level synth synth-loc synth-seq blocked-unlock-pickup unlock-local unlock-pickup
)

module purge
module load python/3.8.2
conda activate hf_gia

python create_babyai_dataset.py --name_env=${ENVS[$SLURM_ARRAY_TASK_ID]} --saving_path="dataset/${NAMES[$SLURM_ARRAY_TASK_ID]}" --max_num_episodes=100000