"""Vendored from k-diffusion v0.1.1 (Katherine Crowson, MIT License)."""

import math

import torch
from torch import nn

from . import utils


class VDenoiser(nn.Module):
    """A v-diffusion-pytorch model wrapper for k-diffusion."""

    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigma):
        return sigma.atan() / math.pi * 2

    def t_to_sigma(self, t):
        return (t * math.pi / 2).tan()

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip
