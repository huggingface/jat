#!/bin/bash
# creates 500,000 per environment from models hosted on the hub

ENVS=(
    atari_alien
    atari_amidar
    atari_assault
    atari_asterix
    atari_asteroid
    atari_atlantis
    atari_bankheist
    atari_battlezone
    atari_beamrider
    atari_berzerk
    atari_bowling
    atari_boxing
    atari_breakout
    atari_centipede
    atari_choppercommand
    atari_crazyclimber
    atari_defender
    atari_demonattack
    atari_doubledunk
    atari_enduro
    atari_fishingderby
    atari_freeway
    atari_frostbite
    atari_gopher
    atari_gravitar
    atari_hero
    atari_icehockey
    atari_jamesbond
    atari_kangaroo
    atari_krull
    atari_kongfumaster
    atari_montezuma
    atari_mspacman
    atari_namethisgame
    atari_phoenix
    atari_pitfall
    atari_pong
    atari_privateye
    atari_qbert
    atari_riverraid
    atari_roadrunner
    atari_robotank
    atari_seaquest
    atari_skiing
    atari_solaris
    atari_spaceinvaders
    atari_stargunner
    atari_surround
    atari_tennis
    atari_timepilot
    atari_tutankham
    atari_upndown
    atari_venture
    atari_videopinball
    atari_wizardofwor
    atari_yarsrevenge
    atari_zaxxon
)


for ENV in "${ENVS[@]}"; do
    python -m sample_factory.huggingface.load_from_hub -r edbeeching/atari_2B_${ENV}_1111 -d train_dir
    echo $ENV
    python data/envs/atari/create_atari_dataset.py --env=$ENV --experiment=atari_2B_${ENV}_1111 --train_dir=train_dir --push_to_hub --hf_repository=edbeeching/prj_jat_dataset_atari_2B_${ENV}_1111 --max_num_frames=500000 --no_render
done