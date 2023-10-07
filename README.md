# stable-audio-tools
Training and inference code for audio generation models

# Install

The library can be installed from PyPI with:
```bash
$ pip install stable-audio-tools
```

To run the training scripts or inference code, you'll want to clone this repository, navigate to the root, and run:
```bash
$ pip install .
```

# Requirements
Requires PyTorch 2.0 or later for Flash Attention support

Development for the repo is done in Python 3.8.10

# Training

## Prerequisites
Before starting your training run, you'll need a model config file, as well as a dataset config file. For more information about those, refer to the Configurations section below

The training code also requires a Weights & Biases account to log the training outputs and demos. Create an account and log in with:
```bash
$ wandb login
```

## Start training
To start a training run, run the `train.py` script in the repo root with:
```bash
$ python3 ./train.py --dataset-config /path/to/dataset/config --model-config /path/to/model/config --name harmonai_train
```

The `--name` parameter will set the project name for your Weights and Biases run.

## Training wrappers and model unwrapping
`stable-audio-tools` uses PyTorch Lightning to facilitate multi-GPU and multi-node training. 

When a model is being trained, it is wrapped in a "training wrapper", which is a `pl.LightningModule` that contains all of the relevant objects needed only for training. That includes things like discriminators for autoencoders, EMA copies of models, and all of the optimizer states.

The checkpoint files created during training include this training wrapper, which greatly increases the size of the checkpoint file.

`unwrap_model.py` in the repo root will take in a wrapped model checkpoint and save a new checkpoint file including only the model itself.

That can be run with from the repo root with:
```bash
$ python3 ./unwrap_model.py --model-config /path/to/model/config --ckpt-path /path/to/wrapped/ckpt --name model_unwrap
```

Unwrapped model checkpoints are required for:
  - Inference scripts
  - Using a model as a pretransform for another model (e.g. using an autoencoder model for latent diffusion)
  - Fine-tuning a pre-trained model with a modified configuration (i.e. partial initialization)

## Fine-tuning
Fine-tuning a model involves continuning a training run from a pre-trained checkpoint. 

To continue a training run from a wrapped model checkpoint, you can pass in the checkpoint path to `train.py` with the `--ckpt-path` flag.

To start a fresh training run using a pre-trained unwrapped model, you can pass in the unwrapped checkpoint to `train.py` with the `--pretrained-ckpt-path` flag.

## Additional training flags

Additional optional flags for `train.py` include:
- `--config-file`
  - The path to the defaults.ini file in the repo root, required if running `train.py` from a directory other than the repo root
- `--pretransform-ckpt-path`
  - Used in various model types such as latent diffusion models to load a pre-trained autoencoder. Requires an unwrapped model checkpoint.
- `--save-dir`
  - The directory in which to save the model checkpoints
- `--checkpoint-every`
  - The number of steps between saved checkpoints.
  - *Default*: 10000
- `--batch-size`
  - Number of samples per-GPU during training. Should be set as large as your GPU VRAM will allow.
  - *Default*: 8
- `--num-gpus`
  - Number of GPUs per-node to use for training
  - *Default*: 1
- `--num-nodes`
  - Number of GPU nodes being used for training
  - *Default*: 1
- `--accum-batches`
  - Enables and sets the number of batches for gradient batch accumulation. Useful for increasing effective batch size when training on smaller GPUs.
- `--strategy`
  - Multi-GPU strategy for distributed training. Setting to `deepspeed` will enable DeepSpeed ZeRO Stage 2.
  - *Default*: `ddp` if `--num_gpus` > 1, else None
- `--precision`
  - floating-point precision to use during training
  - *Default*: 16
- `--num-workers`
  - Number of CPU workers used by the data loader
- `--seed`
  - RNG seed for PyTorch, helps with deterministic training

# Configurations
Training and inference code for `stable-audio-tools` is based around JSON configuration files that define model hyperparameters, training settings, and information about your training dataset.

## Model config
The model config file defines all of the information needed to load a model for training or inference. It also contains the training configuration needed to fine-tune a model or train from scratch.

The following properties are defined in the top level of the model configuration:

- `model_type`
  - The type of model being defined, currently limited to one of `"autoencoder", "diffusion_uncond", "diffusion_cond", "diffusion_cond_inpaint", "diffusion_autoencoder", "musicgen"`.
- `sample_size`
  - The length of the audio provided to the model during training, in samples. For diffusion models, this is also the raw audio sample length used for inference.
- `sample_rate`
  - The sample rate of the audio provided to the model during training, and generated during inference, in Hz.
- `audio_channels`
  - The number of channels of audio provided to the model during training, and generated during inference. Defaults to 2. Set to 1 for mono.
- `model`
  - The specific configuration for the model being defined, varies based on `model_type`
- `training`
  - The training configuration for the model, varies based on `model_type`. Provides parameters for training as well as demos.

## Dataset config
`stable-audio-tools` currently supports two kinds of data sources: local directories of audio files, and WebDataset datasets stored in Amazon S3.

# Todo
- [ ] Add documentation for dataset configs
- [ ] Add documentation for different model types
- [ ] Add documentation on pretransforms
- [ ] Add documentation for Gradio interface
- [ ] Add troubleshooting section
