import torch
import math
import numpy as np
import random

from torch import nn, sin, pow
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchaudio import transforms as T
from alias_free_torch import Activation1d
from typing import List, Literal, Dict, Any, Callable
from einops import rearrange

from ..inference.sampling import sample_v, build_schedule
from ..inference.utils import prepare_audio
from .blocks import SnakeBeta, ResidualUnit
from .bottleneck import Bottleneck, DiscreteBottleneck
from .diffusion import ConditionedDiffusionModel, DAU1DCondWrapper, UNet1DCondWrapper, DiTWrapper
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform, AutoencoderPretransform
from .wavelets import WaveletEncode1d, WaveletDecode1d
from .transformer import ContinuousTransformer, TransformerBlock, RotaryEmbedding, Attention
from .attn_masks import generate_sliding_window_mask, generate_chunked_sliding_window_mask

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        act = Activation1d(act)
    
    return act

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

def create_blocked_mask(audio, block_size, num_blocks):
    shape = audio.shape
    mask = torch.zeros_like(audio)
    for _ in range(num_blocks):
        block_start = torch.randint(0, shape[-1] - block_size, (shape[0],))
        block_end = block_start + block_size
        for i in range(shape[0]):
            mask[i,:, block_start[i]:block_end[i]] = 1
    return mask.bool()


class ResidualDownsampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, conv_bias = True, **kwargs):
        super().__init__()
        self.scale_factor = 1/stride
        self.layer = WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding = math.ceil(stride/2) if stride > 1 else 0, bias=conv_bias)
        self.map = WNConv1d(in_channels, out_channels, 1)
    def forward(self, x):
        if self.scale_factor != 1:
            res = F.interpolate(x, scale_factor=self.scale_factor, mode = 'linear')
            filtered = self.layer(x)
            out = (1.0/math.sqrt(2)) * (self.map(res) + filtered)
        else:
            out = self.map(x)
        return out

class ResidualUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, conv_bias = True, causal=False, **kwargs):
        super().__init__() 
        self.scale_factor = stride
        self.layer = WNConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding = math.ceil(stride/2) if stride > 1 else 0, bias=conv_bias)
        self.map = WNConv1d(in_channels, out_channels, 1)
    def forward(self, x):
        if self.scale_factor != 1:
            res = F.interpolate(x, scale_factor=self.scale_factor, mode = 'linear')
            filtered = self.layer(x)
            out = (1.0/math.sqrt(2)) * (self.map(res) + filtered)
        else:
            out = self.map(x)
        return out


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return rearrange(x, '... a b -> ... b a')

def _zero_pad_modulo_sequence(x, size, dim=-2):
    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
    return x

class TransformerResamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sliding_window = None, chunk_size = 128, chunk_midpoint_shift = False, type = 'encoder', transformer_depth = 3, checkpointing = False,
                 conformer = False, layer_scale = False, dim_heads = 128, differential = True, variable_stride = False, feat_scale = False,
                 sinusoidal_blocks = 0, mask_noise = 0, ff_mult = 3, mapping_bias = True, cross_attn = False, dyt = True, conv_mapping = False, freeze_backbone = False, **kwargs):
        super().__init__()
        if type not in ['encoder', 'decoder']:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")

        self.checkpointing = checkpointing

        transformer_dim = out_channels if type == 'encoder' else in_channels
        transformers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = variable_stride
        self.stride = stride
        self.mapping = WNConv1d(in_channels, out_channels, 3 if conv_mapping else 1, padding = 'same', bias = mapping_bias) if in_channels != out_channels else nn.Identity()
        self.chunk_size = chunk_size
        self.chunk_midpoint_shift = chunk_midpoint_shift
        self.type = type
        self.mask_noise = mask_noise
        self.sliding_window_latents = sliding_window

        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(stride)
        self.transformer_depth = transformer_depth
        for i in range(transformer_depth):
            sinusoidal = True if ((transformer_depth - i) < sinusoidal_blocks) else False
            transformers.append(TransformerBlock(transformer_dim,
                                                 dim_heads = dim_heads,
                                                 causal = False,
                                                 zero_init_branch_outputs = True if not layer_scale else False,
                                                 norm_type = 'dyt' if dyt else 'rms_norm',
                                                 conformer = conformer,
                                                 layer_scale = layer_scale,
                                                 add_rope = True,
                                                 attn_kwargs={'qk_norm': "dyt" if dyt else "rms", "qk_norm_eps": 1e-3, "differential": differential, "feat_scale": feat_scale},
                                                 ff_kwargs={'mult': ff_mult, 'no_bias': False, "sinusoidal": sinusoidal},
                                                 norm_kwargs = {'eps': 1e-3},
                                                 cross_attend = cross_attn))

        self.new_tokens = nn.Parameter(1e-5 * torch.randn(1, self.output_seg_size if not self.variable_stride else 1, out_channels if type == 'encoder' else in_channels))

        self.transformers = nn.ModuleList(transformers)

        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length = 0):
        if window is None:
            return None
        else:
            return [(win * (stride + 1 + prepend_cond_length)) for win in window]

    def _get_seg_sizes(self, stride, prepend_cond_length = 0):
        sub_chunk_size = stride + 1 + prepend_cond_length
        if self.sliding_window_latents is None:
            assert (self.chunk_size % stride) == 0, f"Stride must fit evenly into chunk size:{self.chunk_size}"
        input_seg_size = stride if self.type == 'encoder' else 1
        output_seg_size = 1 if self.type == 'encoder' else stride
        return input_seg_size, output_seg_size, sub_chunk_size

    #@torch.compile
    def forward(self, x, stride = None, return_features = False, override_new_tokens = None, prepend_cond = None, cross_attn_cond = None):
        batch_size = x.shape[0]
        input_length = x.shape[-1]
        if return_features:
            features = []

        if stride == None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            if not self.variable_stride:
                print("cannot override stride if variable_stride is not set")
            prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride, prepend_cond_length)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride, prepend_cond_length)

        if self.type == 'encoder':
            # Pad before mapping so silence zeros get projected through the mapping,
            # rather than inserting raw zeros in the mapped space
            if self.transformer_depth > 0:
                if sliding_window is None:
                    pad_modulo = self.chunk_size
                else:
                    pad_modulo = input_seg_size
                x = _zero_pad_modulo_sequence(x, pad_modulo, dim=-1)
            x = self.mapping(x)

        if self.transformer_depth > 0:
            x = rearrange(x, '... a b -> ... b a')
            if return_features:
                features.append(x)
            if self.type != 'encoder':
                if sliding_window is None:
                    active_stride = stride if stride is not None else self.stride
                    pad_modulo = self.chunk_size // active_stride
                    x = _zero_pad_modulo_sequence(x, pad_modulo)
                else:
                    x = _zero_pad_modulo_sequence(x, input_seg_size)
            x = rearrange(x, 'b (n c) d -> (b n) c d', c = input_seg_size)
            new_token_seq_dim = -1 if not self.variable_stride else output_seg_size
            new_tokens = self.new_tokens.expand([x.shape[0],new_token_seq_dim,-1])
            if override_new_tokens is not None:
                #print(f"Using override new tokens with shape {override_new_tokens.shape}, new tokens shape {new_tokens.shape}, x shape {x.shape}")
                override_new_tokens = rearrange(override_new_tokens, 'b (n c) d -> (b n) c d', c = output_seg_size)
                new_tokens = new_tokens + override_new_tokens
            elif self.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x,new_tokens], dim = -2)
            if prepend_cond is not None:
                n = x.shape[0] // batch_size
                cond_folded = prepend_cond.unsqueeze(1).expand(batch_size, n, prepend_cond.shape[-2], x.shape[-1]).reshape(n * batch_size, prepend_cond.shape[-2], x.shape[-1])
                x = torch.cat([cond_folded, x], dim=-2)   
            x = rearrange(x, '(b n) c d -> b (n c) d', b=batch_size)#.contiguous()

            # Fold into contiguous chunks if no sliding window
            if sliding_window is None:
                prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
                effective_chunk_size = self.chunk_size + self.chunk_size * (1 + prepend_cond_length) // (stride if stride is not None else self.stride)

            if sliding_window is None and self.chunk_midpoint_shift:
                split = self.transformer_depth // 2
                shift = effective_chunk_size // 2

                # First half: standard chunks
                nc = x.shape[1] // effective_chunk_size
                x = rearrange(x, 'b (nc cc) d -> (b nc) cc d', cc=effective_chunk_size)
                cross_attn_first = None
                if cross_attn_cond is not None:
                    cross_attn_first = cross_attn_cond.repeat_interleave(nc, dim=0)
                for layer in self.transformers[:split]:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context = cross_attn_first, self_attention_flash_sliding_window = None)
                    else:
                        x = layer(x, context = cross_attn_first)
                    if return_features:
                        features.append(rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size))
                x = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)

                # Second half: shifted chunks with sub-chunk repeat padding
                # shift is always a multiple of sub_chunk_size, so slicing
                # by shift gives whole sub-chunks in their original order
                x = torch.cat([x[:, :shift, :], x, x[:, -shift:, :]], dim=1)
                nc_shifted = x.shape[1] // effective_chunk_size
                x = rearrange(x, 'b (nc cc) d -> (b nc) cc d', cc=effective_chunk_size)
                cross_attn_second = None
                if cross_attn_cond is not None:
                    cross_attn_second = cross_attn_cond.repeat_interleave(nc_shifted, dim=0)
                for layer in self.transformers[split:]:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context = cross_attn_second, self_attention_flash_sliding_window = None)
                    else:
                        x = layer(x, context = cross_attn_second)
                    if return_features:
                        feat = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)
                        features.append(feat[:, shift:-shift, :])
                x = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)
                x = x[:, shift:-shift, :]
            else:
                if sliding_window is None:
                    x = rearrange(x, 'b (nc cc) d -> (b nc) cc d', cc=effective_chunk_size)

                for layer in self.transformers:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context = cross_attn_cond, self_attention_flash_sliding_window = sliding_window)
                    else:
                        x = layer(x, context = cross_attn_cond, self_attention_flash_sliding_window = sliding_window)
                    if return_features:
                        features.append(x)

                # Unfold chunks back to original batch
                if sliding_window is None:
                    x = rearrange(x, '(b nc) cc d -> b (nc cc) d', b=batch_size)

            x = rearrange(x, 'b (n c) d -> (b n) c d', c=sub_chunk_size)
            x = x[:,-output_seg_size:,:]
            x = rearrange(x, '(b n) c d -> b d (n c)', b=batch_size)
        if self.type == 'decoder':
            x = self.mapping(x)
        if return_features:
            return x, features
        else:   
            return x


class SAMEEncoder(nn.Module):
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 sliding_window = None,
                 checkpointing = False,
                 conformer = False,
                 layer_scale = False,
                 causal = False,
                 differential = True,
                 variable_stride = False,
                 mask_noise = 0.0,
                 conv_mapping = False,
                 freeze_backbone = False,
                 **kwargs
        ):
        super().__init__()
        self.in_channels = in_channels
        self.strides = strides

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [in_channels] + channel_dims

        self.depth = len(c_mults)

        layers = []

        for i in range(self.depth):
            layers += [TransformerResamplingBlock(in_channels=channel_dims[i], out_channels=channel_dims[i+1], stride=strides[i], transformer_depth = transformer_depths[i],
                                                  sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, causal = causal,
                                                  differential = differential, variable_stride = variable_stride, mask_noise = mask_noise, conv_mapping = conv_mapping,
                                                  freeze_backbone = freeze_backbone, **kwargs)]

        layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for param in self.layers[-2].parameters():
                param.requires_grad = False

    def forward(self, x, override_stride = None, return_features = False, **kwargs):
        if override_stride != None:
            assert isinstance(override_stride, list), "override_stride must be a list"
            assert len(override_stride) == self.depth, "override_stride must be a list containing strides for every layer"
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                if override_stride != None:
                    stride = override_stride[i]
                else:
                    stride = None
                if return_features:
                    x, features = layer(x, stride = stride, return_features = True)
                else:
                    x = layer(x, stride = stride)
            else:
                x = layer(x)
        if return_features:
            return x, features
        else:
            return x

class SAMEDecoder(nn.Module):
    def __init__(self,
                 out_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 sliding_window = None,
                 checkpointing = False,
                 conformer = False,
                 layer_scale = False,
                 causal = False,
                 differential = True,
                 variable_stride = False,
                 sinusoidal_blocks = [0,0,0,0],
                 mask_noise = 0.0,
                 conv_mapping = False,
                 freeze_backbone = False,
                 **kwargs
        ):
        super().__init__()

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [out_channels] + channel_dims

        self.depth = len(c_mults)

        layers = [Transpose(), nn.Linear(latent_dim, channel_dims[-1]), Transpose()]

        for i in range(self.depth, 0, -1):
            layers += [TransformerResamplingBlock(in_channels=channel_dims[i], out_channels=channel_dims[i-1], stride=strides[i-1], type = 'decoder', transformer_depth = transformer_depths[i-1],
                                                  sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, causal = causal, differential = differential,
                                                  variable_stride = variable_stride, sinusoidal_blocks = sinusoidal_blocks[i-1], mask_noise = mask_noise, conv_mapping = conv_mapping,
                                                  freeze_backbone = freeze_backbone, **kwargs)]

        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for param in self.layers[1].parameters():
                param.requires_grad = False

    def forward(self, x, override_stride = None, **kwargs):
        if override_stride != None:
            assert isinstance(override_stride, list), "override_stride must be a list"
            assert len(override_stride) == self.depth, "override_stride must be a list containing strides for every layer"

        transformer_layer_index = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                if override_stride != None:
                    stride = override_stride[transformer_layer_index]
                else:
                    stride = None
                x = layer(x, stride = stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x


class TAAEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, type = 'encoder', transformer_depth = 3, use_snake = False, sliding_window = [31,32], checkpointing = False, conformer = False, layer_scale = True, use_dilated_conv = False):
        super().__init__()
        if type not in ['encoder', 'decoder']:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")
        
        self.checkpointing = checkpointing
        
        transformer_dim = out_channels if type == 'encoder' else in_channels
        transformers = []
        transformers.append(Transpose())

        self.sliding_window = sliding_window

        for _ in range(transformer_depth):
            transformers.append(TransformerBlock(transformer_dim, 
                                                 dim_heads = 128, 
                                                 causal = False, 
                                                 zero_init_branch_outputs = True if not layer_scale else False, 
                                                 conformer = conformer, 
                                                 layer_scale = layer_scale, 
                                                 add_rope = True, 
                                                 attn_kwargs={'qk_norm': "ln"}, 
                                                 ff_kwargs={'mult': 4, 'no_bias': False},
                                                 norm_kwargs = {'eps': 1e-2}))
        transformers.append(Transpose())
        transformers = nn.ModuleList(transformers)

        if type == 'encoder':
            layers = []
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake))
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=in_channels))
            layers.append(WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)) if stride > 1 else nn.Identity())
            layers.append(transformers)
            self.layers = nn.ModuleList(layers)
        elif type == 'decoder':
            layers = []
            layers.append(transformers)
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=out_channels))
            layers.append(WNConvTranspose1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)) if stride > 1 else nn.Identity())
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for transformer in layer:
                    if self.checkpointing:
                        x = checkpoint(transformer, x, self_attention_flash_sliding_window = self.sliding_window)
                    else:
                        x = transformer(x, self_attention_flash_sliding_window = self.sliding_window)
            else:
                if self.checkpointing:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
        return x

class TAAEEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 mapping_style = 'conv',
                 **kwargs
        ):
        super().__init__()
        self.in_channels = in_channels
        
        if mapping_style not in ['conv', 'linear', 'none']:
            raise ValueError(f"Unknown mapping style {mapping_style}. Must be 'conv','linear' or 'none")

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims
        if mapping_style == 'none':
            channel_dims[0] = in_channels

        self.depth = len(c_mults)

        if mapping_style == 'conv':
            layers = [WNConv1d(in_channels=in_channels, out_channels=channel_dims[0], kernel_size=7, padding=3, bias = True)]
        elif mapping_style == 'linear':
            layers = [Transpose(), nn.Linear(in_channels, channel_dims[0]), Transpose()]
        elif mapping_style == 'none':
            layers = []

        for i in range(self.depth):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i+1], stride=strides[i], transformer_depth = transformer_depths[i], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]

        if mapping_style == 'conv':
            layers += [
                get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[-1]),
                WNConv1d(in_channels=channel_dims[-1], out_channels=latent_dim, kernel_size=3, padding=1, bias = True)
            ]
        elif mapping_style in ['linear','none']:
            layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class TAAEDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 mapping_style = 'conv',
                 **kwargs
        ):
        super().__init__()
        if mapping_style not in ['conv', 'linear', 'none']:
            raise ValueError(f"Unknown mapping style {mapping_style}. Must be 'conv','linear' or 'none'")

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims
        if mapping_style == 'none':
            channel_dims[0] = out_channels


        self.depth = len(c_mults)

        if mapping_style == 'conv':
            layers = [
                WNConv1d(in_channels=latent_dim, out_channels=channel_dims[-1], kernel_size=3, padding=1, bias = True)
            ]
        elif mapping_style in ['linear', 'none']:
            layers = [Transpose(), nn.Linear(latent_dim, channel_dims[-1]), Transpose()]
        
        for i in range(self.depth, 0, -1):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i-1], stride=strides[i-1], type = 'decoder', transformer_depth = transformer_depths[i-1], 
                                 use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]  

        if mapping_style == 'conv':
            layers += [get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[0]),
                        WNConv1d(in_channels=channel_dims[0], out_channels=out_channels, kernel_size=7, padding=3, bias = False)]
        elif mapping_style == 'linear':
            layers += [Transpose(), nn.Linear(channel_dims[0], out_channels), Transpose()]

            
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.stride = stride

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels, 
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
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
                 use_snake=False,
                 antialias_activation=False
        ):
        super().__init__()
        self.in_channels = in_channels
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
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
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DACEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Encoder as DACEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x

class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Decoder as DACDecoder

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
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False,
        freeze_pretransform = False
    ):
        super().__init__()  
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.freeze_pretransform = freeze_pretransform
        if self.pretransform is not None:
            if self.freeze_pretransform:
                for p in self.pretransform.parameters():
                    p.requires_grad = False
            else:
                for p in self.pretransform.parameters():
                    p.requires_grad = True

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, return_pretransform = False, **kwargs):

        info = {}
        if self.pretransform is not None and not skip_pretransform: 
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)
        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i+1], **kwargs))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio, **kwargs)
        else:
            latents = audio
        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)
        
        if return_info and return_pretransform:
            return latents, info, audio
        elif return_info:
            return latents, info
        elif return_pretransform:
            return latents, audio
        else:
            return latents

    def decode(self, latents, iterate_batch=False, return_loss = False, **kwargs):
        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)
        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1], **kwargs))
            decoded = torch.cat(decoded, dim=0)
        else:
            if return_loss:
                decoded, loss = self.decoder(latents, **kwargs)
            else:
                decoded = self.decoder(latents, **kwargs)
        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        if return_loss:
            return decoded, loss
        else:
            return decoded
          
    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''

        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)
        
    
    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list]*batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape)==2, "Audio should be shape (Channels x Length) with no batch dimension" 
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length, 
                target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor 
        return torch.stack(new_audio) 

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        samples_per_latent = int(self.downsampling_ratio)
        if not chunked or audio.shape[-1] < chunk_size * samples_per_latent:
            return self.encode(audio, **kwargs)

        # chunk_size/overlap are in latent units; scale to samples for slicing.
        chunk_size_samples = chunk_size * samples_per_latent
        hop_samples = (chunk_size - overlap) * samples_per_latent
        total_samples = audio.shape[-1]

        # Anchor the final chunk to the signal end (may overlap more than `overlap`).
        chunk_starts = list(range(0, total_samples - chunk_size_samples + 1, hop_samples))
        if chunk_starts[-1] != total_samples - chunk_size_samples:
            chunk_starts.append(total_samples - chunk_size_samples)

        encoded_chunks = [self.encode(audio[..., s:s + chunk_size_samples]) for s in chunk_starts]

        total_latents = total_samples // samples_per_latent
        half_overlap_latents = overlap // 2
        output = audio.new_zeros(*encoded_chunks[0].shape[:-1], total_latents)
        num_chunks = len(chunk_starts)

        # Trim half the overlap off inner edges (edges facing a neighbour chunk).
        for i, (start_sample, chunk) in enumerate(zip(chunk_starts, encoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_latents - chunk_size) if is_last else (start_sample // samples_per_latent)
            left  = 0 if is_first else half_overlap_latents
            right = chunk_size if is_last else chunk_size - half_overlap_latents
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked or latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)

        # chunk_size/overlap are in latent units — same as `latents`, no scaling.
        samples_per_latent = int(self.downsampling_ratio)
        hop_latents = chunk_size - overlap
        total_latents = latents.shape[-1]

        # Anchor the final chunk to the signal end (may overlap more than `overlap`).
        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        if chunk_starts[-1] != total_latents - chunk_size:
            chunk_starts.append(total_latents - chunk_size)

        decoded_chunks = [self.decode(latents[..., s:s + chunk_size]) for s in chunk_starts]

        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent
        output = latents.new_zeros(*decoded_chunks[0].shape[:-1], total_samples)
        num_chunks = len(chunk_starts)

        # Trim half the overlap off inner edges (edges facing a neighbour chunk).
        for i, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_samples - chunk_size_samples) if is_last else (start_latent * samples_per_latent)
            left  = 0 if is_first else half_overlap_samples
            right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output

    
class DiffusionAutoencoder(AudioAutoencoder):
    def __init__(
        self,
        diffusion: ConditionedDiffusionModel,
        diffusion_downsampling_ratio,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.min_length = self.downsampling_ratio * diffusion_downsampling_ratio

        if self.encoder is not None:
            # Shrink the initial encoder parameters to avoid saturated latents
            with torch.no_grad():
                for param in self.encoder.parameters():
                    param *= 0.5

    def decode(self, latents, steps=100):

        upsampled_length = latents.shape[2] * self.downsampling_ratio

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        if self.decoder is not None:
            latents = self.decode(latents)
    
        # Upsample latents to match diffusion length
        if latents.shape[2] != upsampled_length:
            latents = F.interpolate(latents, size=upsampled_length, mode='nearest')

        noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
        t = build_schedule(steps=steps, include_endpoint=False, device=latents.device)
        decoded = sample_v(self.diffusion, noise, sigmas=t, input_concat_cond=latents)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
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
        from encodec.modules import SEANetEncoder
        seanet_encoder_config = encoder_config["config"]

        #SEANet encoder expects strides in reverse order
        seanet_encoder_config["ratios"] = list(reversed(seanet_encoder_config.get("ratios", [2, 2, 2, 2, 2])))
        encoder = SEANetEncoder(
            **seanet_encoder_config
        )
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]

        encoder = DACEncoderWrapper(**dac_config)
    elif encoder_type == "patched":
        encoder = PatchEncoder(
            **encoder_config["config"]
        )
    elif encoder_type == "staged_patched":
        encoder = StagedPatchEncoder(
            **encoder_config["config"]
        )
    elif encoder_type == "taae":
        encoder = TAAEEncoder(
            **encoder_config["config"]
        )
    elif encoder_type in ("same", "taae_v2"):
        encoder = SAMEEncoder(
            **encoder_config["config"]
        )
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
        from encodec.modules import SEANetDecoder

        decoder = SEANetDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]

        decoder = DACDecoderWrapper(**dac_config)
    elif decoder_type == "patched":
        decoder = PatchDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "staged_patched":
        decoder = StagedPatchDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "taae":
        decoder = TAAEDecoder(
            **decoder_config["config"]
        )
    elif decoder_type in ("same", "taae_v2"):
        decoder = SAMEDecoder(
            **decoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(config: Dict[str, Any]):
    
    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)
    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip,
    )

def create_diffAE_from_config(config: Dict[str, Any]):
    
    diffae_config = config["model"]

    if "encoder" in diffae_config:
        encoder = create_encoder_from_config(diffae_config["encoder"])
    else:
        encoder = None

    if "decoder" in diffae_config:
        decoder = create_decoder_from_config(diffae_config["decoder"])
    else:
        decoder = None

    diffusion_model_type = diffae_config["diffusion"]["type"]

    if diffusion_model_type == "DAU1d":
        diffusion = DAU1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "adp_1d":
        diffusion = UNet1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "dit":
        diffusion = DiTWrapper(**diffae_config["diffusion"]["config"])

    latent_dim = diffae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = diffae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = diffae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    bottleneck = diffae_config.get("bottleneck", None)

    pretransform = diffae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    diffusion_downsampling_ratio = None,

    if diffusion_model_type == "DAU1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["strides"])
    elif diffusion_model_type == "adp_1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["factors"])
    elif diffusion_model_type == "dit":
        diffusion_downsampling_ratio = 1

    return DiffusionAutoencoder(
        encoder=encoder,
        decoder=decoder,
        diffusion=diffusion,
        io_channels=io_channels,
        sample_rate=sample_rate,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        diffusion_downsampling_ratio=diffusion_downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )
