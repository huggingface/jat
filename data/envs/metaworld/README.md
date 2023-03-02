## TODO

- [ ] Improve train and enjoy to be generic to every env
- [ ] Create setup.py with the dependencies (use the branch O.26 of Metaworld)
- [ ] Solve warnings


Command lines:

Train:

```sh
python train.py --env pick-place-v2 --with_wandb=True --wandb_user=qgallouedec --wandb_project sample_facotry_metaworld
```

Push to hub:

```sh
python enjoy.py --algo=APPO --env=pick-place-v2 --experiment=default_experiment --train_dir=./train_dir --max_num_episodes=10 --push_to_hub --hf_repository=qgallouedec/pick-place-v2-sf --save_video --no_render --enjoy_script=enjoy --train_script=train --load_checkpoint_kind best
```

Generate dataset: