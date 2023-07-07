#!/bin/bash

ENVS=(
    assembly
    basketball
    bin-picking
    box-close
    button-press-topdown
    button-press-topdown-wall
    button-press
    button-press-wall
    coffee-button
    coffee-pull
    coffee-push
    dial-turn
    disassemble
    door-close
    door-lock
    door-open
    door-unlock
    drawer-close
    drawer-open
    faucet-close
    faucet-open
    hammer
    hand-insert
    handle-press-side
    handle-press
    handle-pull-side
    handle-pull
    lever-pull
    peg-insert-side
    peg-unplug-side
    pick-out-of-hole
    pick-place
    pick-place-wall
    plate-slide-back-side
    plate-slide-back
    plate-slide-side
    plate-slide
    push-back
    push
    push-wall
    reach
    reach-wall
    shelf-place
    soccer
    stick-pull
    stick-push
    sweep-into
    sweep
    window-close
    window-open
)

for ENV in "${ENVS[@]}"; do
    python -m sample_factory.huggingface.load_from_hub -r qgallouedec/$ENV-v2
    python generate_dataset.py --env $ENV-v2 --experiment $ENV-v2 --train_dir=./train_dir
done
