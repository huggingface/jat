#!/bin/bash
# creates 10k episodes per environment from models hosted on the hub

ENVS=(
    ant halfcheetah hopper doublependulum pendulum reacher swimmer walker
)

for ENV in "${ENVS[@]}"; do
    python -m sample_factory.huggingface.load_from_hub -r edbeeching/mujoco_${ENV}_1111 -d train_dir
    echo $ENV
    python create_mujoco_dataset.py --env=mujoco_${ENV} --experiment=mujoco_${ENV}_1111 --train_dir=train_dir --push_to_hub --max_num_frames=1000000000 --no_render --max_num_episodes=100
done