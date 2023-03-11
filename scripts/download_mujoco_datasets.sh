!/bin/bash

ENVS=(
    ant halfcheetah hopper doublependulum pendulum reacher swimmer walker
)
for ENV in "${ENVS[@]}"; do
    git clone https://huggingface.co/datasets/edbeeching/prj_gia_dataset_mujoco_${ENV}_1111/ data/imitation/mujoco/prj_gia_dataset_mujoco_${ENV}_1111/
done
