import torch
import math
import numpy as np

from torch import nn
from torch.nn import functional as F
from functools import partial
from torch import nn, sin, pow
from torch.nn import Parameter
from alias_free_torch import Activation1d
from encodec.modules import SEANetEncoder, SEANetDecoder
from dac.model.dac import Encoder as DACEncoder, Decoder as DACDecoder
from dac.nn.layers import WNConv1d, WNConvTranspose1d
from typing import Literal, Dict, Any, Callable, Optional

from ..inference.sampling import sample
from .bottleneck import Bottleneck
from .diffusion import create_diffusion_uncond_from_config
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False):
        super().__init__()
        
        self.dilation = dilation

        if use_snake:
            act = partial(Activation1d, activation=SnakeBeta(out_channels))
        else:
            act = nn.ELU

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            act(),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            act(),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False):
        super().__init__()

        if use_snake:
            act = partial(Activation1d, activation=SnakeBeta(in_channels))
        else:
            act = nn.ELU

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            act(),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride//2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False):
        super().__init__()

        if use_snake:
            act = partial(Activation1d, activation=SnakeBeta(in_channels))
        else:
            act = nn.ELU

        self.layers = nn.Sequential(
            act(),
            WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False
        ):
        super().__init__()
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            Activation1d(SnakeBeta(c_mults[-1] * channels)) if use_snake else nn.ELU(),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False):
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i-1]*channels, stride=strides[i-1], use_snake=use_snake)]

        layers += [
            Activation1d(SnakeBeta(c_mults[0] * channels)) if use_snake else nn.ELU(),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DACEncoderWrapper(nn.Module):
    def __init__(self, latent_dim, in_channels=1, **kwargs):
        super().__init__()

        self.encoder = DACEncoder(**kwargs)
        self.latent_dim = latent_dim

        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1)

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x

class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        self.decoder = DACDecoder(**kwargs, input_channel = latent_dim, d_out=out_channels)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)

class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        io_channels=2,
        bottleneck: Bottleneck = None,
        encode_fn: Callable[[torch.Tensor, nn.Module], torch.Tensor] = lambda x, encoder: encoder(x),
        decode_fn: Callable[[torch.Tensor, nn.Module], torch.Tensor] = lambda x, decoder: decoder(x),
        pretransform: Pretransform = None
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio

        self.latent_dim = latent_dim
        self.io_channels = io_channels

        self.bottleneck = bottleneck

        self.encoder = encoder
        self.encode_fn = encode_fn

        self.decoder = decoder
        self.decode_fn = decode_fn

        self.pretransform = pretransform
 
    def encode(self, audio, return_info=False, skip_pretransform=False):

        info = {}

        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    audio = self.pretransform.encode(audio)

        latents = self.encode_fn(audio, self.encoder)

        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True)

            info.update(bottleneck_info)
            
        if return_info:
            return latents, info

        return latents

    def decode(self, latents, **kwargs):

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        decoded = self.decode_fn(latents, self.decoder, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    decoded = self.pretransform.decode(decoded)
        
        return decoded
    
class DiffusionAutoencoder(AudioAutoencoder):
    def __init__(
        self,
        diffusion,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        # Shrink the initial encoder parameters to avoid saturated latents
        with torch.no_grad():
            for param in self.encoder.parameters():
                param *= 0.5

    def decode(self, latents, steps=100):
        
        upsampled_length = latents.shape[2] * self.downsampling_ratio

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        if self.decoder:
            latents = self.decode_fn(latents, self.decoder)

        noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
        decoded = sample(self.diffusion, noise, steps, 0, cond=latents)

        if self.pretransform is not None:
            with torch.no_grad():
                decoded = self.pretransform.decode(decoded)

        return decoded
        
# AE factories

def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
            **encoder_config["config"]
        )
    
    elif encoder_type == "seanet":
        seanet_encoder_config = encoder_config["config"]

        #SEANet encoder expects strides in reverse order
        seanet_encoder_config["ratios"] = list(reversed(seanet_encoder_config.get("ratios", [2, 2, 2, 2, 2])))
        encoder = SEANetEncoder(
            **seanet_encoder_config
        )
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]

        encoder = DACEncoderWrapper(**dac_config)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "seanet":
        decoder = SEANetDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]

        decoder = DACDecoderWrapper(**dac_config)
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(model_config: Dict[str, Any]):
    
    encoder = create_encoder_from_config(model_config["encoder"])
    decoder = create_decoder_from_config(model_config["decoder"])

    bottleneck = model_config.get("bottleneck", None)

    latent_dim = model_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = model_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = model_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)
    
    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )

def create_diffAE_from_config(model_config: Dict[str, Any]):
    
    encoder = create_encoder_from_config(model_config["encoder"])

    decoder = create_decoder_from_config(model_config["decoder"])

    diffusion = create_diffusion_uncond_from_config(model_config["diffusion"])

    latent_dim = model_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = model_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = model_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"

    bottleneck = model_config.get("bottleneck", None)

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    return DiffusionAutoencoder(
        encoder=encoder,
        decoder=decoder,
        diffusion=diffusion,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )