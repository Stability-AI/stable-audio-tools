import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from encodec.modules import SEANetEncoder, SEANetDecoder
from typing import Literal, Dict, Any

def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl



# Modified from https://github.com/wesbz/SoundStream/blob/main/net.py
def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),
        )

    def forward(self, x):
        return self.layers(x)

class AudioEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=64, latent_dim=32, c_mults = [2, 4, 8, 16, 32], strides = [2, 2, 2, 2, 2]):
        super().__init__()
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth-1):
            layers.append(EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i]))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=64, latent_dim=32, c_mults = [2, 4, 8, 16, 32], strides = [2, 2, 2, 2, 2]):
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers.append(DecoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i-1]*channels, stride=strides[i-1]))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        encode_fn = lambda x, encoder: encoder(x),
        decode_fn = lambda z, decoder: decoder(z),
        bottleneck: Literal["vae", "tanh"] = "tanh",
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio

        self.latent_dim = latent_dim

        self.bottleneck = bottleneck

        self.encoder = encoder
        self.encode_fn = encode_fn

        self.decoder = decoder
        self.decode_fn = decode_fn

    def encode(self, audio, return_info=False):

        info = {}

        if self.bottleneck == "vae":
            mean, scale = self.encode_fn(audio, self.encoder).chunk(2, dim=1)
            latents, kl = vae_sample(mean, scale)
            info['kl'] = kl
        else:
            latents = self.encode_fn(audio, self.encoder)
            
            if self.bottleneck == "tanh":
                latents = torch.tanh(latents)
            elif self.bottleneck == "l2_norm":
                latents = F.normalize(latents, dim=1)

        if return_info:
            return latents, info

        return latents

    def decode(self, latents):

        if self.bottleneck == "l2_norm":
            latents = F.normalize(latents, dim=1)

        return self.decode_fn(latents, self.decoder)
    
# AE factories
def create_autoencoder_from_config(config: Dict[str, Any]):
    ae_type = config.get("type", "audio_ae")
    if ae_type == "audio_ae":
        return create_audio_ae_from_config(config["config"])
    elif ae_type == "seanet":
        return create_seanet_ae_from_config(config["config"])
    else:
        raise ValueError(f"Unknown autoencoder type {ae_type}")

def create_audio_ae_from_config(config: Dict[str, Any]):
    strides = config.get("strides", [2, 2, 2, 2, 2])
    
    downsampling_ratio = np.prod(strides)

    latent_dim = config.get("latent_dim", 32)

    bottleneck = config.get("bottleneck", "tanh")

    encoder = AudioEncoder(
        in_channels = config.get("in_channels", 2), 
        channels = config.get("channels", 64),
        latent_dim = latent_dim * 2 if bottleneck == "vae" else latent_dim,
        c_mults = config.get("c_mults", [2, 4, 8, 16, 32]),
        strides = strides
    )

    decoder = AudioDecoder(
        out_channels=config.get("out_channels", 2), 
        channels = config.get("channels", 64),
        latent_dim = latent_dim,
        c_mults = config.get("c_mults", [2, 4, 8, 16, 32]),
        strides = strides
    )

    return AudioAutoencoder(
        encoder,
        decoder,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        bottleneck=bottleneck
    )

def create_seanet_ae_from_config(config):
    
    io_channels = config.get("io_channels", 2)
    latent_dim = config.get("latent_dim", 32)
    bottleneck = config.get("bottleneck", "tanh")
    base_channels = config.get("base_channels", 64)
    n_residual_layers = config.get("n_residual_layers", 3)
    dilation_base = config.get("dilation_base", 3)
    norm = config.get("norm", "none")

    strides = config.get("strides", [2, 2, 2, 2, 2])
    downsampling_ratio = np.prod(strides)
    
    encoder_dim = latent_dim * 2 if bottleneck == 'vae' else latent_dim
    
    encoder = SEANetEncoder(
        channels=io_channels,
        dimension=encoder_dim,
        n_filters=base_channels,
        n_residual_layers=n_residual_layers,
        dilation_base=dilation_base,
        ratios=list(reversed(strides)),
        norm=norm
    )

    decoder = SEANetDecoder(
        channels=io_channels,
        dimension=latent_dim,
        n_filters=base_channels,
        n_residual_layers=n_residual_layers,
        dilation_base=dilation_base,
        ratios=strides,
        norm=norm
    )   

    return AudioAutoencoder(
        encoder,
        decoder,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        bottleneck=bottleneck
    )