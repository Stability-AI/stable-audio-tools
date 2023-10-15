"""k-diffusion transformer diffusion models, version 2."""

from dataclasses import dataclass
from functools import reduce
import math
from typing import Union, List, Literal

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder

from .blocks import FourierFeatures
from .local_attention import ContinuousLocalTransformer

use_compile = False

if use_compile:
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

# Param tags

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Kernels

def compile(function, *args, **kwargs):
    if not use_compile:
        return function
    try:
        return torch.compile(function, *args, **kwargs)
    except RuntimeError:
        return function


@compile
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@compile
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

# Layers

class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)

# Mapping network

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features * self.patch_size, out_features, bias=False)

    def forward(self, x):
        x = rearrange(x, "b (n r) c -> b n (c r)", r=self.patch_size)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features, out_features * self.patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b n (c r) -> b (n r) c", r=self.patch_size)    


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=2):
        super().__init__()
        
        self.patch_size = patch_size
        self.proj = nn.Linear(in_features, out_features * self.patch_size, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "b n (c r) -> b (n r) c", r=self.patch_size)   
        return torch.lerp(skip, x, self.fac.to(x.dtype))

class GlobalAttnTransformerLayer(nn.Module):
    def __init__(self, width, depth, d_heads, mapping_cond_dim=0, cond_token_dim=0, **kwargs):
        super().__init__()
        
        if cond_token_dim > 0:
            self.cond_tokens_proj = nn.Linear(cond_token_dim, width, bias=False)

        if mapping_cond_dim > 0:
            self.mapping_cond_proj = nn.Linear(mapping_cond_dim, width, bias=False)

        self.transformer = ContinuousTransformerWrapper(
                    dim_in=width,
                    dim_out=width,
                    max_seq_len=0, #Not needed without absolute positional embeddings
                    attn_layers = Encoder(
                        dim=width,
                        depth=depth,
                        heads= width // d_heads,
                        attn_flash = True,
                        cross_attend = cond_token_dim > 0,
                        zero_init_branch_output=True,
                        use_abs_pos_emb = False,
                        rotary_pos_emb=True,
                        ff_swish = True,
                        ff_glu = True,
                        **kwargs
                    )
        )

    def forward(self, x, *args, cond=None, cross_attn_cond=None, **kwargs):
        
        if cond is not None:
            cond = self.mapping_cond_proj(cond)

        if cross_attn_cond is not None:
            cross_attn_cond = self.cond_tokens_proj(cross_attn_cond)

        return self.transformer(
            x,
            prepend_embeds=cond.unsqueeze(1),
            context=cross_attn_cond,
            *args,
            **kwargs
        )[:, 1:, :] # Take off the prepended token

# Model class

class HourglassDiffusionTransformer(nn.Module):
    def __init__(
            self, 
            widths: List[int],
            depths: List[int],
            d_heads: List[int],
            io_channels: int,  
            patch_sizes: List[int],
            window_sizes: List[int] = [], # Not needed if use_local_levels = False
            cond_token_dim = 0,
            mapping_cond_dim = 0,
            mapping_dim = 1024,
            mapping_depth = 2,
            mapping_d_ff = 1024,
            use_local_levels = True,
            **kwargs
        ):
        super().__init__()
        self.io_channels = io_channels
        
        self.patch_in = TokenMerge(self.io_channels, widths[0], patch_sizes[0])

        self.mapping_dim = mapping_dim

        self.cond_token_dim = cond_token_dim
        self.mapping_cond_dim = mapping_cond_dim

        self.time_emb = FourierFeatures(1, mapping_dim)
        
        self.mapping_cond_in_proj = nn.Linear(mapping_cond_dim, mapping_dim, bias=False) if mapping_cond_dim else None
        self.time_in_proj = nn.Linear(mapping_dim, mapping_dim, bias=False)

        self.mapping = tag_module(MappingNetwork(mapping_depth, mapping_dim, mapping_d_ff), "mapping")

        assert len(widths) == len(depths) == len(d_heads) == len(patch_sizes), "widths, depths, d_heads, and patch_sizes must have the same length"

        assert use_local_levels == False or (len(window_sizes) == 0 or (len(window_sizes) == len(widths) - 1)), "window_sizes must have one less element than widths"

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(zip(widths, depths, d_heads)):
            
            width, depth, d_heads = spec

            if i < len(widths) - 1:
                if use_local_levels:
                    self.down_levels.append(ContinuousLocalTransformer(
                        dim = width,
                        depth = depth,
                        heads= width // d_heads,
                        use_conv = False,
                        cond_dim = mapping_dim,
                        cross_attn_cond_dim=cond_token_dim,
                        use_rotary_pos_emb = True,
                        local_attn_window_size = window_sizes[i],
                    ))
                    
                    self.up_levels.append(ContinuousLocalTransformer(
                        dim = width,
                        depth = depth,
                        heads = width // d_heads,
                        use_conv = False,
                        cond_dim = mapping_dim,
                        cross_attn_cond_dim=cond_token_dim,
                        local_attn_window_size = window_sizes[i],
                    ))
                else:
                    self.down_levels.append(GlobalAttnTransformerLayer(
                        width=width,
                        depth=depth,
                        d_heads=d_heads,
                        mapping_cond_dim=mapping_dim,
                        cond_token_dim=cond_token_dim,
                        **kwargs
                    ))

                    self.up_levels.append(GlobalAttnTransformerLayer(
                        width=width,
                        depth=depth,
                        d_heads=d_heads,
                        mapping_cond_dim=mapping_dim,
                        cond_token_dim=cond_token_dim,
                        **kwargs
                    ))

            else:
                self.mid_level = GlobalAttnTransformerLayer(
                    width=width,
                    depth=depth,
                    d_heads=d_heads,
                    mapping_cond_dim=mapping_dim,
                    cond_token_dim=cond_token_dim,
                    **kwargs
                )

       
        self.merges = nn.ModuleList([TokenMerge(spec_1, spec_2, ps) for spec_1, spec_2, ps in zip(widths[:-1], widths[1:], patch_sizes[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2, spec_1, ps) for spec_1, spec_2, ps in zip(widths[:-1], widths[1:], patch_sizes[1:])])
       
        self.out_norm = RMSNorm(widths[0])
        self.patch_out = TokenSplitWithoutSkip(widths[0], io_channels, patch_sizes[0])
        nn.init.zeros_(self.patch_out.proj.weight)
        
    def _forward(self, x, t, cond_tokens=None, cond_tokens_mask=None, mapping_cond=None):
     
        # Patching
        x = rearrange(x, "b c t -> b t c")

        x = self.patch_in(x)

        # Mapping network 
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        time_emb = self.time_in_proj(self.time_emb(t[..., None]))
        
        mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        
        cond = self.mapping(time_emb + mapping_emb)

        # Hourglass transformer
        skips = []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, cond=cond, cross_attn_cond=cond_tokens)
            skips.append(x)
            x = merge(x)
                                
        x = self.mid_level(x, cond=cond, cross_attn_cond=cond_tokens)

        for up_level, split, skip in reversed(list(zip(self.up_levels, self.splits, skips))):
            x = split(x, skip)
            x = up_level(x, cond=cond, cross_attn_cond=cond_tokens)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = rearrange(x, "b t c -> b c t")

        return x

    def forward(self, 
                x, 
                t, 
                cond_tokens=None, 
                cond_tokens_mask=None, 
                mapping_cond=None, 
                cfg_dropout_prob=0.0, 
                cfg_scale=1.0,
                rescale_cfg=False,
                scale_phi = 1.0):
     
        if self.cond_token_dim > 0 or self.mapping_cond_dim > 0:
            
            # CFG dropout
            if cfg_dropout_prob > 0.0:
                if cond_tokens is not None:
                    null_embed = torch.zeros_like(cond_tokens, device=cond_tokens.device)
                    dropout_mask = torch.bernoulli(torch.full((cond_tokens.shape[0], 1, 1), cfg_dropout_prob, device=cond_tokens.device)).to(torch.bool)
                    cond_tokens = torch.where(dropout_mask, null_embed, cond_tokens)

                if mapping_cond is not None:
                    null_embed = torch.zeros_like(mapping_cond, device=mapping_cond.device)
                    dropout_mask = torch.bernoulli(torch.full((mapping_cond.shape[0], 1, 1), cfg_dropout_prob, device=mapping_cond.device)).to(torch.bool)
                    mapping_cond = torch.where(dropout_mask, null_embed, mapping_cond)

            if cfg_scale != 1.0:
                # Classifier-free guidance
                # Concatenate conditioned and unconditioned inputs on the batch dimension            
                batch_inputs = torch.cat([x, x], dim=0)
                batch_t = torch.cat([t, t], dim=0)
                
                batch_mapping = torch.cat([mapping_cond, torch.zeros_like(mapping_cond)], dim=0) if mapping_cond is not None else None
                batch_cond_tokens = torch.cat([cond_tokens, torch.zeros_like(cond_tokens, device=cond_tokens.device)], dim=0) if cond_tokens is not None else None
                # if cond_tokens_mask is not None:
                #     batch_masks = torch.cat([cond_tokens_mask, cond_tokens_mask], dim=0).to(torch.bool)
                # else:
                #     batch_masks = None
                
                output = self._forward(batch_inputs, batch_t, cond_tokens=batch_cond_tokens, cond_tokens_mask=cond_tokens_mask, mapping_cond=batch_mapping)

                cond_output, uncond_output = torch.chunk(output, 2, dim=0)
                return uncond_output + (cond_output - uncond_output) * cfg_scale

            else:

                return self._forward(x, t, cond_tokens=cond_tokens, cond_tokens_mask=cond_tokens_mask, mapping_cond=mapping_cond)

        else:

            return self._forward(x, t)
