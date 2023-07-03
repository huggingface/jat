# General Intelligent Agents

An open impementation of [GATO](https://www.deepmind.com/publications/a-generalist-agent)


dev install  (Linux)
`pip install -e .[dev]`

Steps:

1. Creation of imitation learning data on the hub filter datasets for `prj_gia*`
2. Creation of tokenizer, model, training loop etc
3. Single env learning, e.g all Atari envs -> evaluation
4. Multi task learning, e.g Atari + MassiveText -> evaluation
5. Full tasks setting -> evaluation  <- we are here

More details to come!

## Usage

### Dataset loading

See [GIA Dataset](https://huggingface.co/datasets/gia-project/gia-dataset)

### GIA model

TODO: write this section

### Training
A training script is provided: [train.py](scripts/train.py).

As this script relies on HuggingFace's [Trainer](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer), [TrainingArguments](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.TrainingArguments) can be passed to the script. For example, if one wants to have a test loss computed on 64 samples of each task every 100 training steps::
```bash
python script/train.py --output_dir=./output \
                       --num_train_epochs=4 \
                       --preprocessing_num_workers=16 \  # number of parallel processes to use when preprocessing the datasets
                       --evaluation_strategy=steps \  # perform evaluation on test datasets every n *steps*
                       --prediction_loss_only=True \  # ask Trainer to only compute prediction loss to optimize evaluation
                       --eval_steps=100 \  # evaluate every 100 *steps* (depending on evaluation_strategy)
                       --max_eval_samples=64 \  # number of test samples to compute the loss on
                       --per_device_train_batch_size=8 \  # training batch size per GPU/TPU core/CPU 
                       --per_device_eval_batch_size=8  # evaluation batch size per GPU/TPU core/CPU 
```

