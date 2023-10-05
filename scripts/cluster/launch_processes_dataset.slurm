#!/bin/bash
#SBATCH --job-name=gia_proc   # name of job
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err

source /fsx/ed/miniconda3/etc/profile.d/conda.sh

conda activate gia
#TASKS=
TASKS=$1
#atari-alien,atari-amidar,atari-assault,atari-asterix,atari-asteroids,atari-atlantis,atari-bankheist,atari-battlezone,atari-beamrider,atari-berzerk,atari-bowling,atari-boxing,atari-breakout,atari-centipede,atari-choppercommand,atari-crazyclimber,atari-defender,atari-demonattack,atari-doubledunk,atari-fishingderby,atari-freeway,atari-frostbite,atari-gravitar,atari-hero,atari-icehockey,atari-jamesbond,atari-kangaroo,atari-krull,atari-kungfumaster,atari-montezumarevenge,atari-mspacman,atari-namethisgame,atari-phoenix,atari-pitfall,atari-pong,atari-privateeye,atari-qbert,atari-riverraid,atari-roadrunner,atari-robotank,atari-seaquest,atari-skiing,atari-solaris,atari-spaceinvaders,atari-stargunner,atari-surround,atari-tennis,atari-timepilot,atari-tutankham,atari-upndown,atari-venture,atari-videopinball,atari-wizardofwor,atari-yarsrevenge,atari-zaxxon
python scripts/tokenize_dataset.py --output_dir=runs/proc_dataset --task_names=$TASKS --preprocessing_num_worker=16