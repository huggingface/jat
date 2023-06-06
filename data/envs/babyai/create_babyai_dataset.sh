#!/bin/bash
# generates 100,000 episodes per environment using the hand-scripted optimal bot

ENVS=(
    BabyAI-GoTo-v0 BabyAI-GoToDoor-v0 BabyAI-GoToImpUnlock-v0 BabyAI-GoToLocal-v0 BabyAI-GoToObj-v0 BabyAI-GoToObjDoor-v0 BabyAI-GoToRedBall-v0 BabyAI-GoToRedBallGrey-v0 BabyAI-GoToRedBallNoDists-v0 BabyAI-GoToRedBlueBall-v0 BabyAI-GoToSeq-v0 BabyAI-Open-v0 BabyAI-OpenDoor-v0 BabyAI-OpenDoorsOrder-v0 BabyAI-OpenRedDoor-v0 BabyAI-OpenTwoDoors-v0 BabyAI-ActionObjDoor-v0 BabyAI-FindObjS5-v0 BabyAI-KeyCorridor-v0 BabyAI-MoveTwoAcross-v0 BabyAI-OneRoomS8-v0 BabyAI-Pickup-v0 BabyAI-PickupAbove-v0 BabyAI-PickupDist-v0 BabyAI-PickupLoc-v0 BabyAI-UnblockPickup-v0 BabyAI-BossLevel-v0 BabyAI-BossLevelNoUnlock-v0 BabyAI-MiniBossLevel-v0 BabyAI-Synth-v0 BabyAI-SynthLoc-v0 BabyAI-SynthSeq-v0 BabyAI-BlockedUnlockPickup-v0 BabyAI-KeyInBox-v0 BabyAI-Unlock-v0 BabyAI-UnlockLocal-v0 BabyAI-UnlockPickup-v0 BabyAI-UnlockToUnlock-v0
)

NAMES=(
    go-to go-to-door go-to-imp-unlock go-to-local go-to-obj go-to-obj-door go-to-red-ball go-to-red-ball-grey go-to-red-ball-no-dists go-to-red-blue-ball go-to-seq open open-door open-doors-order open-red-door open-two-doors action-obj-door find-obj-s5 key-corridor move-two-across one-room-s8 pickup pickup-above pickup-dist pickup-loc unblock-pickup boss-level boss-level-no-unlock mini-boss-level synth synth-loc synth-seq blocked-unlock-pickup key-in-box unlock unlock-local unlock-pickup unlock-to-unlock
)

for i in "${!ENVS[@]}"; do
    echo "${ENVS[$i]}"
    python create_babyai_dataset.py --name_env=${ENVS[$i]} --saving_path=${NAMES[$i]} --max_num_frames=100000
done