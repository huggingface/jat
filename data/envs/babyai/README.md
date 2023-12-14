# How to create the BabyAI dataset and push it to the hub

## Install
1. Install the jat lib from the root dir `pip install .[dev]`
2. Install Minigrid:
```shell
pip install -r requirements.txt
```

## Launch
The new Minigrid environment provides 38 BabyAI tasks. To generate 100k episodes per task using the Bot agent, use:
```shell
./create_babyai_dataset.sh
```