# How to create the Mujoco dataset and push it to the hub

1. Install the jat lib from the root dir `pip install .[dev]`
2. install the additional dependencies for generating the atari dataset. `pip install -r requirements.txt`
3. python -m sample_factory.huggingface.load_from_hub -r edbeeching/atari_2B_atari_pong_1111 -d train_dir
4. For a single env run `python create_atari_dataset.py --env=atari_pong --experiment=atari_2B_atari_pong_1111 --train_dir=train_dir --push_to_hub --hf_repository=edbeeching/prj_jat_dataset_atari_2B_atari_pong_1111 --max_num_frames=100000 --no_render`
