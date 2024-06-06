# Jack of All Trades, Master of Some, a Multi-Purpose Transformer Agent

[![Build](https://github.com/huggingface/jat/actions/workflows/test-ci.yml/badge.svg?branch=main)](https://github.com/huggingface/jat/actions/workflows/ci.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/huggingface/jat.svg?color=blue)](https://github.com/huggingface/jat/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/cs.AI-arXiv%3A2402.09844-B31B1B.svg)](https://arxiv.org/abs/2402.09844)
[![Website](https://img.shields.io/website/http/huggingface.co/jat-project.svg?down_color=red&down_message=offline&up_message=online)](https://huggingface.co/jat-project)

<p align="center">
  <picture>
    <img alt="Rendering" src="https://github.com/huggingface/gia/assets/45557362/5b4d4920-fafd-4cb8-90d1-ac4df3a97073" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>


## Installation

To get started with JAT, follow these steps:

1. Clone this repository onto your local machine.

    ```shell
    git clone https://github.com/huggingface/jat.git
    cd jat
    ```

2. Create a new virtual environment and activate it, and install required dependencies via pip.

    ```shell
    python3 -m venv env
    source env/bin/activate
    # all deps
    pip install .[dev]
    # training deps
    pip install .[train]
    # eval deps
    pip install .[eval]

    ```

## Demonstration of the trained agent
The trained JAT agent is available [here](https://huggingface.co/jat-project/jat). The following script gives an example of the use of this agent on the Pong environment

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from jat.eval.rl import make

# Load the model and the processor
model_name_or_path = "jat-project/jat"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to("cuda")

# Make the environment
env = make("atari-pong", render_mode="human")

observation, info = env.reset()
reward = None
done = False
model.reset_rl()  # clear key-value cache
while not done:
    action = model.get_next_action(processor, **observation, reward=reward, action_space=env.action_space)
    observation, reward, termined, truncated, info = env.step(action)
    done = termined or truncated

    if done and "episode" not in info:  # handle "fake done" for atari
        observation, info = env.reset()
        done = False

env.close()
```

## Usage Examples

Here are some examples of how you might use JAT in both evaluation and fine-tuning modes. More detailed information about each example is provided within the corresponding script files.
- **Evaluating JAT**: Evaluate pretrained JAT models on specific downstream tasks

    ```shell
    python scripts/eval_jat.py --model_name_or_path jat-project/jat --tasks atari-pong --trust_remote_code
    ```

- **Training JAT**: Train your own JAT model from scratch (run on 8xA100)
    ```shell
    accelerate launch scripts/train_jat_tokenized.py \
    --output_dir checkpoints/jat \
    --model_name_or_path jat-project/jat \
    --tasks all \
    --trust_remote_code \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 2 \
    --save_steps 10000 \
    --run_name train_jat_small \
    --logging_steps 100 \
    --logging_first_step \
    --dispatch_batches False \
    --dataloader_num_workers 16 \
    --max_steps 250000 \
    ```

For further details regarding usage, consult the documentation included with individual script files.

## Dataset
You can find the training dataset used to train the JAT model at this [Hugging Face dataset repo](https://huggingface.co/datasets/jat-project/jat-dataset). The dataset contains a large selection of Reinforcement Learning, textual and multimodal tasks:

**Reinforment Learning tasks**
- Atari 57
- BabyAI
- Meta-World
- MuJoCo

**Textual tasks**
- Wikipedia
- OSCAR

**Visual Question answering tasks**
- OK VQA
- Conceptual Captions

Usage:
```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("jat-project/jat-dataset", "metaworld-assembly")
>>> first_episode = dataset["train"][0]
>>> first_episode.keys()
dict_keys(['continuous_observations', 'continuous_actions', 'rewards'])
>>> len(first_episode["rewards"])
500
>>> first_episode["continuous_actions"][0]
[6.459120273590088, 2.2422609329223633, -5.914587020874023, -19.799840927124023]
```

Check out the dataset's model card for more information.


## Contributing & Issues

We welcome contributions from the community of expert policies, datasets or code improvements.
Feel free to fork the repository and make a PR with your improvements. If you find any problems running the code, please open an issue.

## Citation

Please ensure proper citations when incorporating this work into your projects.

```bibtex
@article{gallouedec2024jack,
    title = {{Jack of All Trades, Master of Some, a Multi-Purpose Transformer Agent}},
    author = {Gallouédec, Quentin and Beeching, Edward and Romac, Clément and Dellandréa, Emmanuel},
    journal = {arXiv preprint arXiv:2402.09844},
    year = {2024},
    url = {https://arxiv.org/abs/2402.09844}
}
```
