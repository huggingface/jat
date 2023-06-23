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
echo "LOCAL RANK" $LOCAL_RANK
echo "PMI_SIZE" $PMI_SIZE
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "MV2_COMM_WORLD_SIZE" $MV2_COMM_WORLD_SIZE
echo "WORLD_SIZE" $WORLD_SIZE

unset LOCAL_RANK
unset PMI_SIZE
unset OMPI_COMM_WORLD_SIZE
unset MV2_COMM_WORLD_SIZE
unset WORLD_SIZE
# export LOCAL_RANK=-1 # workaround for a bug with torch distributed
python /fsx/ed/work/gia/scripts/evaluate.py $1