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


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class RoundFSQ(Module):
    def __init__(self, latent_dim: int, dither: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.dither = dither
        # no dither at inference
        # you control dither at training

    def quantize(self, z):
        if self.training and self.dither:
            z = z + (torch.rand_like(z) - 0.5)
        quantized = round_ste(z)

        return quantized

    @autocast(device_type="cuda", enabled = False)
    def forward(self, z, skip_tanh: bool = False):
        # z: [B, N, T]
        assert z.shape[-1] == self.latent_dim, f"Expected last dim={self.latent_dim}, got {z.shape[-1]}"
        q = self.quantize(z)
        indices = q.to(torch.int64)
        return q, indices
