# Autoencoders
At a high level, autoencoders are models constructed of two parts: an *encoder*, and a *decoder*. 

The *encoder* takes in an sequence (such as mono or stereo audio) and outputs a compressed representation of that sequence as a d-channel "latent sequence", usually heavily downsampled by a constant factor.

The *decoder* takes in a d-channel latent sequence and upsamples it back to the original input sequence length, reversing the compression of the encoder.

Autoencoders are trained with a combination of reconstruction and adversarial losses in order to create a compact and invertible representation of raw audio data that allows downstream models to work in a data-compressed "latent space", with various desirable and controllable properties such as reduced sequence length, noise resistance, and discretization.

The autoencoder architectures defined in `stable-audio-tools` are largely fully-convolutional, which allows autoencoders trained on small lengths to be applied to arbitrary-length sequences. For example, an autoencoder trained on 1-second samples could be used to encode 45-second inputs to a latent diffusion model.

# Model configs
The model config file for an autoencoder should set the `model_type` to `autoencoder`, and the `model` object should have the following properties:

- `encoder`
    - Configuration for the autoencoder's encoder half
- `decoder`
    - Configuration for the autoencoder's decoder half
- `latent_dim`
    - Latent dimension of the autoencoder, used by inference scripts and downstream models
- `downsampling_ratio`
    - Downsampling ratio between the input sequence and the latent sequence, used by inference scripts and downstream models
- `io_channels`
    - Number of input and output channels for the autoencoder when they're the same, used by inference scripts and downstream models
- `bottleneck`
    - Configuration for the autoencoder's bottleneck
    - Optional
- `pretransform`
    - A pretransform definition for the autoencoder, such as wavelet decomposition or another autoencoder
    - See [pretransforms.md](pretransforms.md) for more information
    - Optional
- `in_channels`
    - Specifies the number of input channels for the autoencoder, when it's different from `io_channels`, such as in a mono-to-stereo model
    - Optional
- `out_channels`
    - Specifies the number of output channels for the autoencoder, when it's different from `io_channels`
    - Optional

# Training configs
The `training` config in the autoencoder model config file should have the following properties:
- `learning_rate`
    - The learning rate to use during training
- `use_ema`
    - If true, a copy of the model weights is maintained during training and updated as an exponential moving average of the trained model's weights. 
    - Optional. Default: `false`
- `warmup_steps`
    - The number of training steps before turning on adversarial losses
    - Optional. Default: `0`
- `encoder_freeze_on_warmup`
    - If true, freezes the encoder after the warmup steps have completed, so adversarial training only affects the decoder.
    - Optional. Default: `false`
- `loss_configs`
    - Configurations for the loss function calculation
    - Optional
- `optimizer_configs`
    - Configuration for optimizers and schedulers
    - Optional

## Loss configs
There are few different types of losses that are used for autoencoder training, including spectral losses, time-domain losses, adversarial losses, and bottleneck-specific losses.

Hyperparameters fo these losses as well as loss weighting factors can be configured in the `loss_configs` property in the `training` config.

### Spectral losses
Multi-resolution STFT losses are the main reconstruction loss used for our audio autoencoders. We use the [auraloss](https://github.com/csteinmetz1/auraloss/tree/main/auraloss) library for our spectral loss functions. 

For mono autoencoders (`io_channels` == 1), we use the [MultiResolutionSTFTLoss](https://github.com/csteinmetz1/auraloss/blob/1576b0cd6e927abc002b23cf3bfc455b660f663c/auraloss/freq.py#L329) module. 

For stereo autoencoders (`io_channels` == 2), we use the [SumAndDifferenceSTFTLoss](https://github.com/csteinmetz1/auraloss/blob/1576b0cd6e927abc002b23cf3bfc455b660f663c/auraloss/freq.py#L533) module. 

#### Example config
```json
"spectral": {
    "type": "mrstft",
    "config": {
        "fft_sizes": [2048, 1024, 512, 256, 128, 64, 32],
        "hop_sizes": [512, 256, 128, 64, 32, 16, 8],
        "win_lengths": [2048, 1024, 512, 256, 128, 64, 32],
        "perceptual_weighting": true
    },
    "weights": {
        "mrstft": 1.0
    }
}
```

### Time-domain loss
We compute the L1 distance between the original audio and the decoded audio to provide a time-domain loss.

#### Example config
```json
"time": {
    "type": "l1",
    "weights": {
        "l1": 0.1
    }
}
```

### Adversarial losses
Adversarial losses bring in an ensemble of discriminator models to discriminate between real and fake audio, providing a signal to the autoencoder on perceptual discrepancies to fix.

We largely rely on the [multi-scale STFT discriminator](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/msstftd.py#L99) from the EnCodec repo

#### Example config
```json
"discriminator": {
    "type": "encodec",
    "config": {
        "filters": 32,
        "n_ffts": [2048, 1024, 512, 256, 128],
        "hop_lengths": [512, 256, 128, 64, 32],
        "win_lengths": [2048, 1024, 512, 256, 128]
    },
    "weights": {
        "adversarial": 0.1,
        "feature_matching": 5.0
    }
}
```

## Demo config
The only property to set for autoencoder training demos is the `demo_every` property, determining the number of steps between demos.

### Example config
```json
"demo": {
    "demo_every": 2000
}
```

# Encoder and decoder types
Encoders and decoders are defined separately in the model configuration, so encoders and decoders from different model architectures and libraries can be used interchangeably. 

## Oobleck
Oobleck is Harmonai's in-house autoencoder architecture, implementing features from a variety of other autoencoder architectures.

### Example config
```json
"encoder": {
    "type": "oobleck",
    "config": {
        "in_channels": 2,
        "channels": 128,
        "c_mults": [1, 2, 4, 8],
        "strides": [2, 4, 8, 8],
        "latent_dim": 128,
        "use_snake": true
    }
},
"decoder": {
    "type": "oobleck",
    "config": {
        "out_channels": 2,
        "channels": 128,
        "c_mults": [1, 2, 4, 8],
        "strides": [2, 4, 8, 8],
        "latent_dim": 64,
        "use_snake": true,
        "use_nearest_upsample": false
    }
}
```

## DAC
This is the Encoder and Decoder definitions from the `descript-audio-codec` repo. It's a simple fully-convolutional autoencoder with channels doubling every level. The encoder and decoder configs are passed directly into the constructors for the DAC [Encoder](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L64) and [Decoder](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/dac.py#L115).

**Note: This does not include the DAC quantizer, and does not load pre-trained DAC models, this is just the encoder and decoder definitions.**

### Example config
```json
"encoder": {
    "type": "dac",
    "config": {
        "in_channels": 2,
        "latent_dim": 32,
        "d_model": 128,
        "strides": [2, 4, 4, 4]
    }
},
"decoder": {
    "type": "dac",
    "config": {
        "out_channels": 2,
        "latent_dim": 32,
        "channels": 1536,
        "rates": [4, 4, 4, 2]
    }
}
```

## SEANet
This is the SEANetEncoder and SEANetDecoder definitions from Meta's EnCodec repo. This is the same encoder and decoder architecture used in the EnCodec models used in MusicGen, without the quantizer.

The encoder and decoder configs are passed directly into the [SEANetEncoder](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/modules/seanet.py#L66C12-L66C12) and [SEANetDecoder](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/modules/seanet.py#L147) classes directly, though we reverse the input order of the strides (ratios) in the encoder to make it consistent with the order in the decoder.

### Example config
```json
"encoder": {
    "type": "seanet",
    "config": {
        "channels": 2,
        "dimension": 128,
        "n_filters": 64,
        "ratios": [4, 4, 8, 8],
        "n_residual_layers": 1,
        "dilation_base": 2,
        "lstm": 2,
        "norm": "weight_norm"
    }
},
"decoder": {
    "type": "seanet",
    "config": {
        "channels": 2,
        "dimension": 64,
        "n_filters": 64,
        "ratios": [4, 4, 8, 8],
        "n_residual_layers": 1,
        "dilation_base": 2,
        "lstm": 2,
        "norm": "weight_norm"
    }
},
```

# Bottlenecks
In our terminology, the "bottleneck" of an autoencoder is a module placed between the encoder and decoder to enforce particular constraints on the latent space the encoder creates.

Bottlenecks have a similar interface to the autoencoder with `encode()` and `decode()` functions defined. Some bottlenecks return extra information in addition to the output latent series, such as quantized token indices, or additional losses to be considered during training.

To define a bottleneck for the autoencoder, you can provide the `bottleneck` object in the autoencoder's model configuration, with the following 

## VAE

The Variational Autoencoder (VAE) bottleneck splits the encoder's output in half along the channel dimension, treats the two halves as the "mean" and "scale" parameters for VAE sampling, and performs the latent sampling. At a basic level, the "scale" values determine the amount of noise to add to the "mean" latents, which creates a noise-resistant latent space where more of the latent space decodes to perceptually "valid" audio. This is particularly helpful for diffusion models where the outpus of the diffusion sampling process leave a bit of Gaussian error noise.

**Note: For the VAE bottleneck to work, the output dimension of the encoder must be twice the size of the input dimension for the decoder.**

### Example config
```json
"bottleneck": {
    "type": "vae"
}
```

### Extra info
The VAE bottleneck also returns a `kl` value in the encoder info. This is the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between encoded/sampled latent space and a Gaussian distribution. By including this value as a loss value to optimize, we push our latent distribution closer to a normal distribution, potentially trading off reconstruction quality.

### Example loss config
```json
"bottleneck": {
    "type": "kl",
    "weights": {
        "kl": 1e-4
    }
}
```

## Tanh
This bottleneck applies the tanh function to the latent series, "soft-clipping" the latent values to be between -1 and 1. This is a quick and dirty way to enforce a limit on the variance of the latent space, but training these models can be unstable as it's seemingly easy for the latent space to saturate the values to -1 or 1 and never recover.

### Example config
```json
"bottleneck": {
    "type": "tanh"
}
```

## Wasserstein
The Wasserstein bottleneck implements the WAE-MMD regularization method from the [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) paper, calculating the Maximum Mean Discrepancy (MMD) between the latent space and a Gaussian distribution. Including this value as a loss value to optimize leads to a more Gaussian latent space, but does not require stochastic sampling as with a VAE, so the encoder is deterministic.

The Wasserstein bottleneck also exposes the `noise_augment_dim` property, which concatenates `noise_augment_dim` channels of Gaussian noise to the latent series before passing into the decoder. This adds some stochasticity to the latents which can be helpful for adversarial training, while keeping the encoder outputs deterministic.

**Note: The MMD calculation is very VRAM-intensive for longer sequence lengths, so training a Wasserstein autoencoder is best done on autoencoders with a decent downsampling factor, or on short sequence lengths. For inference, the MMD calculation is disabled.**

### Example config
```json
"bottleneck": {
    "type": "wasserstein"
}
```

### Extra info
This bottleneck adds the `mmd` value to the encoder info, representing the Maximum Mean Discrepancy.

### Example loss config
```json
"bottleneck": {
    "type": "mmd",
    "weights": {
        "mmd": 100
    }
}
```

## L2 normalization (Spherical autoencoder)
The L2 normalization bottleneck normalizes the latents across the channel-dimension, projecting the latents to a d-dimensional hypersphere. This acts as a form of latent space normalization.


### Example config
```json
"bottleneck": {
    "type": "l2_norm"
}
```


## RVQ
Residual vector quantization (RVQ) is currently the leading method for learning discrete neural audio codecs (tokenizers for audio). In vector quantization, each item in the latent sequence is individually "snapped" to the nearest vector in a discrete "codebook" of learned vectors. The index of the vector in the codebook can then be used as a token index for things like autoregressive transformers. Residual vector quantization improves the precision of normal vector quantization by adding additional codebooks. For a deeper dive into RVQ, check out [this blog post by Dr. Scott Hawley](https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html).

This RVQ bottleneck uses [lucidrains' implementation](https://github.com/lucidrains/vector-quantize-pytorch/tree/master) from the `vector-quantize-pytorch` repo, which provides a lot of different quantizer options. The bottleneck config is passed through to the `ResidualVQ`  [constructor](https://github.com/lucidrains/vector-quantize-pytorch/blob/0c6cea24ce68510b607f2c9997e766d9d55c085b/vector_quantize_pytorch/residual_vq.py#L26).

**Note: This RVQ implementation uses manual replacement of codebook vectors to reduce codebook collapse. This does not work with multi-GPU training as the random replacement is not synchronized across devices.** 

### Example config
```json
"bottleneck": {
    "type": "rvq",
    "config": {
        "num_quantizers": 4,
        "codebook_size": 2048,
        "dim": 1024,
        "decay": 0.99,
    }
}
```

## DAC RVQ
This is the residual vector quantization implementation from the `descript-audio-codec` repo. It differs from the above implementation in that it does not use manual replacements to improve codebook usage, but instead uses learnable linear layers to project the latents down to a lower-dimensional space before performing the individual quantization operations. This means it's compatible with distributed training. 

The bottleneck config is passed directly into the `ResidualVectorQuantize` [constructor](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py#L97).

The `quantize_on_decode` property is also exposed, which moves the quantization process to the decoder. This should not be used during training, but is helpful when training latent diffusion models that use the quantization process as a way to remove error after the diffusion sampling process.

### Example config
```json
"bottleneck": {
    "type": "dac_rvq",
    "config": {
        "input_dim": 64,
        "n_codebooks": 9,
        "codebook_dim": 32,
        "codebook_size": 1024,
        "quantizer_dropout": 0.5
    }
}
```

### Extra info
The DAC RVQ bottleneck also adds the following properties to the `info` object:
- `pre_quantizer`
    - The pre-quantization latent series, useful in combination with `quantize_on_decode` for training latent diffusion models.
- `vq/commitment_loss`
    - Commitment loss for the quantizer
- `vq/codebook_loss`
    - Codebook loss for the quantizer
