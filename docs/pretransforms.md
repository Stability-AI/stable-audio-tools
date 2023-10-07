# Pretransforms
Many models require some fixed transform to be applied to the input audio before the audio is passed in to the trainable layers of the model, as well as a corresponding inverse transform to be applied to the outputs of the model. We refer to these as "pretransforms".

At the moment, `stable-audio-tools` supports two pretransforms, frozen autoencoders for latent diffusion models and wavelet decompositions.

Pretransforms have a similar interface to autoencoders with "encode" and "decode" functions defined for each pretransform.

## Autoencoder pretransform
To define a model with an autoencoder pretransform, you can define the "pretransform" property in the model config, with the `type` property set to `autoencoder`. The `config` property should be an autoencoder model definition.

Example:
```json
"pretransform": {
    "type": "autoencoder",
    "config": {
        "encoder": {
            ...
        },
        "decoder": {
            ...
        }
        ...normal autoencoder configuration
    }
}
```

### Latent rescaling
The original [Latent Diffusion paper](https://arxiv.org/abs/2112.10752) found that rescaling the latent series to unit variance before performing diffusion improved quality. To this end, we expose a `scale` property on autoencoder pretransforms that will take care of this rescaling. The scale should be set to the original standard deviation of the latents, which can be determined experimentally, or by looking at the `latent_std` value during training. The pretransform code will divide by this scale factor in the `encode` function and multiply by this scale in the `decode` function.

## Wavelet pretransform
`stable-audio-tools` also exposes wavelet decomposition as a pretransform. Wavelet decomposition is a quick way to trade off sequence length for channels in autoencoders, while maintaining a multi-band implicit bias.

Wavelet pretransforms take the following properties:

- `channels`
    - The number of input and output audio channels for the wavelet transform
- `levels`
    - The number of successive wavelet decompositions to perform. Each level doubles the channel count and halves the sequence length
- `wavelet`
    - The specific wavelet from [PyWavelets](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html) to use, currently limited to `"bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior4.4", "bior6.8"`

## Future work
We hope to add more filters and transforms to this list, including PQMF and STFT transforms.