# Metaworld dataset

## Installation

```sh
pip install -r requirements.txt
```

## Train

```sh
./train_all.sh
```

Push to hub:

```sh
python enjoy.py --algo=APPO --env=pick-place-v2 --experiment=pick-place-v2 --train_dir=./train_dir --max_num_episodes=10 --push_to_hub --hf_repository=qgallouedec/pick-place-v2-sf --save_video --no_render --enjoy_script=enjoy --train_script=train --load_checkpoint_kind best
```

Generate dataset:
