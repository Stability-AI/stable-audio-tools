# Diffusion

Diffusion models learn to denoise data

# Model configs
The model config file for a diffusion model should set the `model_type` to `diffusion_cond` if the model uses conditioning, or `diffusion_uncond` if it does not, and the `model` object should have the following properties:

- `diffusion`
    - The configuration for the diffusion model itself. See below for more information on the diffusion model config
- `pretransform`
    - The configuration of the diffusion model's [pretransform](pretransforms.md), such as an autoencoder for latent diffusion.
    - Optional
- `conditioning`
    - The configuration of the various [conditioning](conditioning.md) modules for the diffusion model
    - Only required for `diffusion_cond`
- `io_channels`
    - The base number of input/output channels for the diffusion model
    - Used by inference scripts to determine the shape of the noise to generate for the diffusion model

# Diffusion configs
- `type`
    - The underlying model type for the transformer
    - For conditioned diffusion models, be one of `dit` ([Diffusion Transformer](#diffusion-transformers-dit)), `DAU1d` ([Dance Diffusion U-Net](#dance-diffusion-u-net)), or `adp_cfg_1d` ([audio-diffusion-pytorch U-Net](#audio-diffusion-pytorch-u-net-adp))
    - Unconditioned diffusion models can also use `adp_1d`
- `cross_attention_cond_ids`
    - Conditioner ids for conditioning information to be used as cross-attention input
    - If multiple ids are specified, the conditioning tensors will be concatenated along the sequence dimension
- `global_cond_ids`
    - Conditioner ids for conditioning information to be used as global conditioning input
    - If multiple ids are specified, the conditioning tensors will be concatenated along the channel dimension
- `prepend_cond_ids`
    - Conditioner ids for conditioning information to be prepended to the model input
    - If multiple ids are specified, the conditioning tensors will be concatenated along the sequence dimension
    - Only works with diffusion transformer models
- `input_concat_ids`
    - Conditioner ids for conditioning information to be concatenated to the model input
    - If multiple ids are specified, the conditioning tensors will be concatenated along the channel dimension
    - If the conditioning tensors are not the same length as the model input, they will be interpolated along the sequence dimension to be the same length.
        - The interpolation algorithm is model-dependent, but usually uses nearest-neighbor resampling.
- `config`
    - The configuration for the model backbone itself
    - Model-dependent

# Training configs
The `training` config in the diffusion model config file should have the following properties:

- `learning_rate`
    - The learning rate to use during training
    - Defaults to constant learning rate, can be overridden with `optimizer_configs`
- `use_ema`
    - If true, a copy of the model weights is maintained during training and updated as an exponential moving average of the trained model's weights. 
    - Optional. Default: `true`
- `log_loss_info`
    - If true, additional diffusion loss info will be gathered across all GPUs and displayed during training
    - Optional. Default: `false`
- `loss_configs`
    - Configurations for the loss function calculation
    - Optional
- `optimizer_configs`
    - Configuration for optimizers and schedulers
    - Optional, overrides `learning_rate`
- `demo`
    - Configuration for the demos during training, including conditioning information

## Example config
```json
"training": {
    "use_ema": true,
    "log_loss_info": false,
    "optimizer_configs": {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": 5e-5,
                    "betas": [0.9, 0.999],
                    "weight_decay": 1e-3
                }
            },
            "scheduler": {
                "type": "InverseLR",
                "config": {
                    "inv_gamma": 1000000,
                    "power": 0.5,
                    "warmup": 0.99
                }
            }
        }
    },
    "demo": { ... }
}
```

# Demo configs
The `demo` config in the diffusion model training config should have the following properties:
- `demo_every`
    - How many training steps between demos
- `demo_steps`
    - Number of diffusion timesteps to run for the demos
- `num_demos`
    - This is the number of examples to generate in each demo
- `demo_cond`
    - For conditioned diffusion models, this is the conditioning metadata to provide to each example, provided as a list
    - NOTE: List must be the same length as `num_demos`
- `demo_cfg_scales`
    - For conditioned diffusion models, this provides a list of classifier-free guidance (CFG) scales to render during the demos. This can be helpful to get an idea of how the model responds to different conditioning strengths as training continues.

## Example config
```json
"demo": {
    "demo_every": 2000,
    "demo_steps": 250,
    "num_demos": 4,
    "demo_cond": [
        {"prompt": "A beautiful piano arpeggio", "seconds_start": 0, "seconds_total": 80},
        {"prompt": "A tropical house track with upbeat melodies, a driving bassline, and cheery vibes", "seconds_start": 0, "seconds_total": 250},
        {"prompt": "A cool 80s glam rock song with driving drums and distorted guitars", "seconds_start": 0, "seconds_total": 180},
        {"prompt": "A grand orchestral arrangement", "seconds_start": 0, "seconds_total": 190}
    ],
    "demo_cfg_scales": [3, 6, 9]
}
```

# Model types

A variety of different model types can be used as the underlying backbone for a diffusion model. At the moment, this includes variants of U-Net and Transformer models.

## Diffusion Transformers (DiT)

Transformers tend to consistently outperform U-Nets in terms of model quality, but are much more memory- and compute-intensive and work best on shorter sequences such as latent encodings of audio.

### Continuous Transformer

This is our custom implementation of a transformer model, based on the `x-transformers` implementation, but with efficiency improvements such as fused QKV layers, and Flash Attention 2 support.

### `x-transformers`

This model type uses the `ContinuousTransformerWrapper` class from the https://github.com/lucidrains/x-transformers repository as the diffusion transformer backbone.

`x-transformers` is a great baseline transformer implementation with lots of options for various experimental settings.
It's great for testing out experimental features without implementing them yourself, but the implementations might not be fully optimized, and breaking changes may be introduced without much warning. 

## Diffusion U-Net

U-Nets use a hierarchical architecture to gradually downsample the input data before more heavy processing is performed, then upsample the data again, using skip connections to pass data across the downsampling "valley" (the "U" in the name) to the upsampling layer at the same resolution. 

### audio-diffusion-pytorch U-Net (ADP)

This model type uses a modified implementation of the `UNetCFG1D` class from version 0.0.94 of the `https://github.com/archinetai/audio-diffusion-pytorch` repo, with added Flash Attention support.

### Dance Diffusion U-Net

This is a reimplementation of the U-Net used in [Dance Diffusion](https://github.com/Harmonai-org/sample-generator). It has minimal conditioning support, only really supporting global conditioning. Mostly used for unconditional diffusion models.