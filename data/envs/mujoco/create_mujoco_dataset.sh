#!/bin/bash
# creates 100,000 per environment from models hosted on the hub

ENVS=(
    ant halfcheetah hopper doublependulum pendulum reacher swimmer walker
)

for ENV in "${ENVS[@]}"; do
    python -m sample_factory.huggingface.load_from_hub -r edbeeching/mujoco_${ENV}_1111 -d train_dir
    echo $ENV
    python create_mujoco_dataset.py --env=mujoco_${ENV} --experiment=mujoco_${ENV}_1111 --train_dir=train_dir --push_to_hub --hf_repository=edbeeching/prj_gia_dataset_mujoco_${ENV}_1111 --max_num_frames=100000 --no_render
done