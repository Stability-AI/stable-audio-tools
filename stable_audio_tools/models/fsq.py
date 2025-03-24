"""
Dithered Finite Scalar Quantization
Code adapted from https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/finite_scalar_quantization.py
"""

from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange


def leaky_hard_clip(x: Tensor, alpha: float = 1e-3) -> Tensor: 
    return (1-alpha) * torch.clamp(x, -1, 1) + alpha * x

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class DitheredFSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dither_inference: bool = False,
        num_codebooks: int = 1,
        noise_dropout: float = 0.5,
        scale: float = 1.0,
    ):
        super().__init__()
        self.levels = levels

        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer("_basis", _basis, persistent = False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.codebook_size = _levels.prod().item()

        self.num_codebooks = num_codebooks

        self.dim = codebook_dim * num_codebooks

        self.dither_inference = dither_inference

        self.scale = scale

        half_l = self.scale * 2 / (self._levels - 1)
        self.register_buffer("half_l", half_l, persistent = False)

        self.allowed_dtypes = (torch.float32, torch.float64)

        self.noise_dropout = noise_dropout

    def quantize(self, z, skip_tanh: bool = False):
        if not skip_tanh: z = torch.tanh(z)

        if not self.training:
            quantized = self._scale_and_shift_inverse(round_ste(self._scale_and_shift(z)))
        else:
            quantized = z
            mask = torch.bernoulli(torch.full([z.shape[0],1,1,1], self.noise_dropout, device = z.device)).bool().expand_as(z)
            quantized = torch.where(mask, quantized, self._scale_and_shift_inverse(round_ste(self._scale_and_shift(quantized))))
            mask = torch.bernoulli(torch.full([z.shape[0],1,1,1], self.noise_dropout, device = z.device)).bool().expand_as(z)
            quantized = torch.where(mask, quantized, z + (torch.rand_like(z) - 0.5) * self.half_l)

        return quantized

    def _scale_and_shift(self, z):
        level_indices = (z + 1 * self.scale) / self.half_l
        return level_indices
    
    def _scale_and_shift_inverse(self, level_indices):
        z = level_indices * self.half_l - 1 * self.scale
        return z

    def _indices_to_codes(self, indices):
        level_indices = self._indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def _codes_to_indices(self, zhat):
        zhat = self._scale_and_shift(zhat)
        zhat = zhat.round().to(torch.int64)
        out = (zhat * self._basis).sum(dim=-1)
        return out

    def _indices_to_level_indices(self, indices):
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        # Expects input of batch x sequence x num_codebooks
        assert indices.shape[-1] == self.num_codebooks, f'expected last dimension of {self.num_codebooks} but found last dimension of {indices.shape[-1]}'
        codes = self._indices_to_codes(indices.to(torch.int64))
        codes = rearrange(codes, '... c d -> ... (c d)')
        return codes

    @autocast(device_type="cuda", enabled = False)
    def forward(self, z, skip_tanh: bool = False):

        orig_dtype = z.dtype

        assert z.shape[-1] == self.dim, f'expected dimension of {self.num_codebooks * self.dim} but found dimension of {z.shape[-1]}'

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # make sure allowed dtype before quantizing

        if z.dtype not in self.allowed_dtypes:
            z = z.to(torch.float64)

        codes = self.quantize(z, skip_tanh=skip_tanh)
        indices = self._codes_to_indices(codes)
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # cast codes back to original dtype

        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        # return quantized output and indices

        return codes, indices
